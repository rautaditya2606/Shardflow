"""
CUDA Graph Runner for ShardFlow Pipeline Nodes.

Captures and replays static computation graphs for transformer layer slices,
eliminating PyTorch CPU kernel launch overhead and memory management jitter.

Supports dual-shape captured graphs:
1. Decode Graph: shape [1, 1, hidden_dim] for single-token autoregressive generation
2. Verify Graph: shape [1, spec_k, hidden_dim] for speculative multi-token verification
"""

import logging
from typing import Optional, Tuple
import torch
import torch.nn as nn
from transformers.cache_utils import StaticCache

logger = logging.getLogger(__name__)


class CUDAGraphRunner:
    """
    Manages CUDA Graph capture and fast replay for a ModelSlice.
    """

    def __init__(
        self,
        layers: nn.ModuleList,
        hidden_size: int,
        device: torch.device,
        dtype: torch.dtype,
        spec_k: int = 4,
        rotary_emb: Optional[nn.Module] = None,
        enabled: bool = True,
    ):
        self.layers = layers
        self.hidden_size = hidden_size
        self.device = torch.device(device)
        self.dtype = dtype
        self.spec_k = spec_k
        self.rotary_emb = rotary_emb
        self.enabled = enabled and (self.device.type == "cuda" and torch.cuda.is_available())

        # Graphs
        self._decode_graph: Optional[torch.cuda.CUDAGraph] = None
        self._verify_graph: Optional[torch.cuda.CUDAGraph] = None

        # Static IO Buffers for Decode [1, 1, D]
        self._static_decode_in: Optional[torch.Tensor] = None
        self._static_decode_out: Optional[torch.Tensor] = None
        self._static_decode_pos: Optional[torch.Tensor] = None
        self._static_decode_cos: Optional[torch.Tensor] = None
        self._static_decode_sin: Optional[torch.Tensor] = None
        # ponytail: cache_position must be a static in-place buffer so graph replay
        # writes to the correct KV slot — derived tensors freeze at capture position 0
        self._static_decode_cache_pos: Optional[torch.Tensor] = None

        # Static IO Buffers for Verify [1, K, D]
        self._static_verify_in: Optional[torch.Tensor] = None
        self._static_verify_out: Optional[torch.Tensor] = None
        self._static_verify_pos: Optional[torch.Tensor] = None
        self._static_verify_cos: Optional[torch.Tensor] = None
        self._static_verify_sin: Optional[torch.Tensor] = None
        self._static_verify_cache_pos: Optional[torch.Tensor] = None

        # Static Cache dedicated to graph capture
        self._capture_cache: Optional[StaticCache] = None
        self.is_captured = False

    def can_use_graph(self, seq_len: int) -> bool:
        """Return True if a graph exists for this sequence length."""
        if not self.enabled or not self.is_captured:
            return False
        return seq_len == 1 or seq_len == (self.spec_k + 1)

    def capture(self, static_cache: StaticCache) -> bool:
        """
        Capture CUDA Graphs for [1, 1, D] and [1, K+1, D] forward passes.
        """
        if not self.enabled:
            logger.info("CUDA Graphs disabled or running on non-CUDA device.")
            return False

        try:
            logger.info("Capturing CUDA Graphs on %s (hidden_size=%d, spec_k=%d)...", self.device, self.hidden_size, self.spec_k)
            self._capture_cache = static_cache

            # 1. Capture Decode Graph [1, 1, D]
            self._static_decode_in = torch.zeros((1, 1, self.hidden_size), device=self.device, dtype=self.dtype)
            self._static_decode_pos = torch.zeros((1, 1), device=self.device, dtype=torch.long)
            # Static cache_position buffer — must be updated in-place before every replay
            self._static_decode_cache_pos = torch.zeros((1,), device=self.device, dtype=torch.long)

            if self.rotary_emb is not None:
                try:
                    pos_emb = self.rotary_emb(self._static_decode_in, self._static_decode_pos)
                    if isinstance(pos_emb, (tuple, list)) and len(pos_emb) == 2:
                        self._static_decode_cos = torch.zeros_like(pos_emb[0])
                        self._static_decode_sin = torch.zeros_like(pos_emb[1])
                except Exception as e:
                    logger.debug("rotary_emb static buffer init exception: %s", e)

            # Warmup on side stream
            s = torch.cuda.Stream(device=self.device)
            s.wait_stream(torch.cuda.current_stream(device=self.device))
            with torch.cuda.stream(s):
                for _ in range(3):
                    if self._static_decode_cos is not None and self.rotary_emb is not None:
                        cos, sin = self.rotary_emb(self._static_decode_in, self._static_decode_pos)
                        self._static_decode_cos.copy_(cos)
                        self._static_decode_sin.copy_(sin)
                    self._static_decode_out = self._forward_eager(
                        self._static_decode_in,
                        self._static_decode_pos,
                        self._capture_cache,
                        cache_position=self._static_decode_cache_pos,
                    )
            torch.cuda.current_stream(device=self.device).wait_stream(s)

            # Capture Decode Graph
            self._decode_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._decode_graph, stream=s):
                self._static_decode_out = self._forward_eager(
                    self._static_decode_in,
                    self._static_decode_pos,
                    self._capture_cache,
                    cache_position=self._static_decode_cache_pos,
                )

            # 2. Capture Verify Graph [1, K+1, D]
            verify_len = self.spec_k + 1
            self._static_verify_in = torch.zeros((1, verify_len, self.hidden_size), device=self.device, dtype=self.dtype)
            self._static_verify_pos = torch.arange(verify_len, device=self.device, dtype=torch.long).unsqueeze(0)
            self._static_verify_cache_pos = torch.arange(verify_len, device=self.device, dtype=torch.long)

            if self.rotary_emb is not None:
                try:
                    pos_emb = self.rotary_emb(self._static_verify_in, self._static_verify_pos)
                    if isinstance(pos_emb, (tuple, list)) and len(pos_emb) == 2:
                        self._static_verify_cos = torch.zeros_like(pos_emb[0])
                        self._static_verify_sin = torch.zeros_like(pos_emb[1])
                except Exception as e:
                    logger.debug("verify rotary_emb static buffer init exception: %s", e)

            with torch.cuda.stream(s):
                for _ in range(3):
                    if self._static_verify_cos is not None and self.rotary_emb is not None:
                        cos, sin = self.rotary_emb(self._static_verify_in, self._static_verify_pos)
                        self._static_verify_cos.copy_(cos)
                        self._static_verify_sin.copy_(sin)
                    self._static_verify_out = self._forward_eager(
                        self._static_verify_in,
                        self._static_verify_pos,
                        self._capture_cache,
                        cache_position=self._static_verify_cache_pos,
                    )
            torch.cuda.current_stream(device=self.device).wait_stream(s)

            self._verify_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._verify_graph, stream=s):
                self._static_verify_out = self._forward_eager(
                    self._static_verify_in,
                    self._static_verify_pos,
                    self._capture_cache,
                    cache_position=self._static_verify_cache_pos,
                )

            self.is_captured = True
            logger.info("CUDA Graphs successfully captured for decode [1, 1, D] and verify [1, %d, D]!", verify_len)
            return True

        except Exception as e:
            logger.warning("CUDA Graph capture failed (falling back to eager execution): %s", e)
            self.is_captured = False
            self._decode_graph = None
            self._verify_graph = None
            return False

    def replay_decode(self, hidden_states: torch.Tensor, position: int) -> torch.Tensor:
        """Replay captured decode graph for [1, 1, D] with dynamic RoPE positioning."""
        if not self.is_captured or self._decode_graph is None:
            raise RuntimeError("Decode graph not captured")

        self._static_decode_in.copy_(hidden_states)
        self._static_decode_pos.fill_(position)
        # ponytail: update cache_position in-place so KV writes go to correct slot
        self._static_decode_cache_pos.fill_(position)

        # Update static RoPE cos/sin buffers dynamically before graph replay
        if self._static_decode_cos is not None and self.rotary_emb is not None:
            cos, sin = self.rotary_emb(self._static_decode_in, self._static_decode_pos)
            self._static_decode_cos.copy_(cos)
            self._static_decode_sin.copy_(sin)

        self._decode_graph.replay()
        return self._static_decode_out

    def replay_verify(self, hidden_states: torch.Tensor, start_position: int) -> torch.Tensor:
        """Replay captured verify graph for [1, K+1, D] with dynamic RoPE positioning."""
        if not self.is_captured or self._verify_graph is None:
            raise RuntimeError("Verify graph not captured")

        verify_len = hidden_states.shape[1]
        self._static_verify_in.copy_(hidden_states)
        for i in range(verify_len):
            self._static_verify_pos[0, i] = start_position + i
            # ponytail: update cache_position in-place so verify KV writes land correctly
            self._static_verify_cache_pos[i] = start_position + i

        if self._static_verify_cos is not None and self.rotary_emb is not None:
            cos, sin = self.rotary_emb(self._static_verify_in, self._static_verify_pos)
            self._static_verify_cos.copy_(cos)
            self._static_verify_sin.copy_(sin)

        self._verify_graph.replay()
        return self._static_verify_out

    def _forward_eager(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        cache: Optional[StaticCache],
        cache_position: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Internal eager forward pass used during graph capture."""
        h = hidden_states
        seq_len = hidden_states.shape[1]

        # Use the static position embeddings buffers if available
        pos_emb = None
        if seq_len == 1 and self._static_decode_cos is not None and self._static_decode_sin is not None:
            pos_emb = (self._static_decode_cos, self._static_decode_sin)
        elif seq_len == self.spec_k and self._static_verify_cos is not None and self._static_verify_sin is not None:
            pos_emb = (self._static_verify_cos, self._static_verify_sin)
        elif self.rotary_emb is not None:
            try:
                pos_emb = self.rotary_emb(h, position_ids)
            except Exception:
                pos_emb = None

        # ponytail: use the passed-in static cache_position buffer (updated in-place before replay)
        # so the graph writes KV entries to the correct advancing slot — never derive from position_ids
        if cache_position is None:
            cache_position = position_ids.squeeze(0)

        for layer in self.layers:
            kwargs = {
                "position_ids": position_ids,
                "past_key_values": cache,
                "use_cache": True,
                "cache_position": cache_position,
            }
            if pos_emb is not None:
                kwargs["position_embeddings"] = pos_emb

            out = layer(h, **kwargs)
            h = out[0] if isinstance(out, tuple) else out

        return h
