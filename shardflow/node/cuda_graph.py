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
        self._static_decode_pos_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        # Static IO Buffers for Verify [1, K, D]
        self._static_verify_in: Optional[torch.Tensor] = None
        self._static_verify_out: Optional[torch.Tensor] = None
        self._static_verify_pos: Optional[torch.Tensor] = None
        self._static_verify_pos_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        # Static Cache dedicated to graph capture
        self._capture_cache: Optional[StaticCache] = None
        self.is_captured = False

    def can_use_graph(self, seq_len: int) -> bool:
        """Return True if a graph exists for this sequence length."""
        if not self.enabled or not self.is_captured:
            return False
        return seq_len == 1 or seq_len == self.spec_k

    def capture(self, static_cache: StaticCache) -> bool:
        """
        Capture CUDA Graphs for [1, 1, D] and [1, K, D] forward passes.
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
            
            # Warmup on side stream
            s = torch.cuda.Stream(device=self.device)
            s.wait_stream(torch.cuda.current_stream(device=self.device))
            with torch.cuda.stream(s):
                for _ in range(3):
                    self._static_decode_out = self._forward_eager(
                        self._static_decode_in,
                        self._static_decode_pos,
                        self._capture_cache,
                    )
            torch.cuda.current_stream(device=self.device).wait_stream(s)

            # Capture Decode Graph
            self._decode_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._decode_graph, stream=s):
                self._static_decode_out = self._forward_eager(
                    self._static_decode_in,
                    self._static_decode_pos,
                    self._capture_cache,
                )

            # 2. Capture Verify Graph [1, K, D]
            self._static_verify_in = torch.zeros((1, self.spec_k, self.hidden_size), device=self.device, dtype=self.dtype)
            self._static_verify_pos = torch.arange(self.spec_k, device=self.device, dtype=torch.long).unsqueeze(0)

            with torch.cuda.stream(s):
                for _ in range(3):
                    self._static_verify_out = self._forward_eager(
                        self._static_verify_in,
                        self._static_verify_pos,
                        self._capture_cache,
                    )
            torch.cuda.current_stream(device=self.device).wait_stream(s)

            self._verify_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._verify_graph, stream=s):
                self._static_verify_out = self._forward_eager(
                    self._static_verify_in,
                    self._static_verify_pos,
                    self._capture_cache,
                )

            self.is_captured = True
            logger.info("CUDA Graphs successfully captured for decode [1, 1, D] and verify [1, %d, D]!", self.spec_k)
            return True

        except Exception as e:
            logger.warning("CUDA Graph capture failed (falling back to eager execution): %s", e)
            self.is_captured = False
            self._decode_graph = None
            self._verify_graph = None
            return False

    def replay_decode(self, hidden_states: torch.Tensor, position: int) -> torch.Tensor:
        """Replay captured decode graph for [1, 1, D]."""
        if not self.is_captured or self._decode_graph is None:
            raise RuntimeError("Decode graph not captured")

        self._static_decode_in.copy_(hidden_states)
        self._static_decode_pos.fill_(position)
        self._decode_graph.replay()
        return self._static_decode_out

    def replay_verify(self, hidden_states: torch.Tensor, start_position: int) -> torch.Tensor:
        """Replay captured verify graph for [1, K, D]."""
        if not self.is_captured or self._verify_graph is None:
            raise RuntimeError("Verify graph not captured")

        self._static_verify_in.copy_(hidden_states)
        for i in range(self.spec_k):
            self._static_verify_pos[0, i] = start_position + i
        self._verify_graph.replay()
        return self._static_verify_out

    def _forward_eager(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        cache: Optional[StaticCache],
    ) -> torch.Tensor:
        """Internal eager forward pass used during graph capture."""
        h = hidden_states
        pos_emb = None
        if self.rotary_emb is not None:
            try:
                pos_emb = self.rotary_emb(h, position_ids)
            except Exception:
                pos_emb = None

        for layer in self.layers:
            kwargs = {
                "position_ids": position_ids,
                "past_key_values": cache,
                "use_cache": True,
            }
            if pos_emb is not None:
                kwargs["position_embeddings"] = pos_emb

            out = layer(h, **kwargs)
            h = out[0] if isinstance(out, tuple) else out

        return h
