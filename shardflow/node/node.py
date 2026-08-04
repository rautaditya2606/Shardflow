"""
Pipeline node — holds a slice of transformer layers, processes activations.

Each node:
1. Loads its assigned layer range on startup
2. Listens for incoming activations via TCP
3. Runs hidden states through its layers
4. Forwards result to the next node (or returns logits if final node)

Phase 1: No KV cache — full sequence recomputed each token.
"""

import argparse
import asyncio
import logging
import sys
from typing import Optional

import torch

from shardflow.node.layer_loader import load_layer_slice, ModelSlice
from shardflow.node.kv_cache import KVCacheStore
from shardflow.orchestrator.sampler import sample_next_token
from shardflow.transport.connection import NodeServer, NodeClient
from shardflow.transport.protocol import (
    MessageType,
    TensorMessage,
)
from transformers.cache_utils import DynamicCache

logger = logging.getLogger(__name__)


class PipelineNode:
    """
    A single node in the inference pipeline.

    Holds a contiguous range of transformer layers and optionally
    the final norm + LM head (if this is the last node).
    Maintains a per-session KV cache.
    """

    def __init__(
        self,
        model_slice: ModelSlice,
        is_first_node: bool = False,
        is_last_node: bool = False,
        next_node_host: Optional[str] = None,
        next_node_port: Optional[int] = None,
        listen_host: str = "0.0.0.0",
        listen_port: int = 9000,
        kv_timeout: float = 60.0,
        max_sessions: int = 32,
    ):
        self.model_slice = model_slice
        self.is_first_node = is_first_node
        self.is_last_node = is_last_node
        self.next_node_host = next_node_host
        self.next_node_port = next_node_port
        self.listen_host = listen_host
        self.listen_port = listen_port

        # KV Cache Store
        self.kv_store = KVCacheStore(eviction_timeout=kv_timeout, max_sessions=max_sessions)

        # Connection to next node (if not the last)
        self._next_client: Optional[NodeClient] = None
        self._server: Optional[NodeServer] = None

    async def start(self) -> None:
        """Start the node: connect to next node, start listening, start eviction loop."""
        await self.kv_store.start_eviction_loop()

        # Connect to the next node in the chain (if not last)
        if not self.is_last_node and self.next_node_host:
            self._next_client = NodeClient(
                self.next_node_host,
                self.next_node_port,
            )
            await self._next_client.connect()
            logger.info(
                "Connected to next node at %s:%d",
                self.next_node_host, self.next_node_port
            )

        # Start TCP server for incoming activations
        self._server = NodeServer(
            host=self.listen_host,
            port=self.listen_port,
            handler=self._handle_message,
        )
        await self._server.start()

        logger.info(
            "Node ready — layers [%d, %d), %s, listening on %s:%d",
            self.model_slice.layer_start,
            self.model_slice.layer_end,
            "LAST node (has LM head)" if self.is_last_node else "INTERMEDIATE node",
            self.listen_host,
            self.listen_port,
        )

    async def _handle_message(self, msg: TensorMessage) -> Optional[TensorMessage]:
        """
        Process an incoming message.

        For CLEAR: evict KV cache for session_id.
        For ACTIVATION:
        - Run hidden states through local layers using cached KV
        - If last node: return sampled token ID (or logits if requested)
        - If not last: forward to next node, wait for response, pass back
        """
        if msg.msg_type == MessageType.CLEAR:
            self.kv_store.evict(msg.session_id)
            if not self.is_last_node and self._next_client and self._next_client.is_connected:
                try:
                    await self._next_client.send(msg)
                except (ConnectionError, OSError):
                    logger.debug("Could not forward CLEAR — next node already disconnected")
            return None

        if msg.msg_type != MessageType.ACTIVATION:
            logger.warning("Unknown message type: %s", msg.msg_type)
            return None

        # Process activation
        hidden_states = msg.tensor.to(self.model_slice.device, non_blocking=True)
        output = self._forward(hidden_states, session_id=msg.session_id)

        if self.is_last_node:
            if msg.sample_on_node:
                logits = output[0, -1, :]
                token_id = sample_next_token(
                    logits,
                    temperature=msg.temperature,
                    top_k=msg.top_k,
                    top_p=msg.top_p,
                )
                return TensorMessage(
                    msg_type=MessageType.TOKEN_ID,
                    session_id=msg.session_id,
                    token_id=token_id,
                )
            else:
                return TensorMessage(
                    msg_type=MessageType.LOGITS,
                    session_id=msg.session_id,
                    tensor=output.cpu(),
                )
        else:
            forward_msg = TensorMessage(
                msg_type=MessageType.ACTIVATION,
                session_id=msg.session_id,
                tensor=output.cpu(),
                temperature=msg.temperature,
                top_k=msg.top_k,
                top_p=msg.top_p,
                sample_on_node=msg.sample_on_node,
            )
            response = await self._next_client.send_recv(forward_msg)
            return response

    @torch.inference_mode()
    def _forward(self, hidden_states: torch.Tensor, session_id: str) -> torch.Tensor:
        """
        Run hidden states through this node's layers with KV caching.

        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            session_id: unique session ID for KV cache lookup

        Returns:
            If last node: logits [batch, seq_len, vocab_size]
            Otherwise: hidden_states [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device

        # Lookup or initialize KV cache for this session
        cache = self.kv_store.get(session_id)
        if cache is None:
            cache = DynamicCache()
            self.kv_store.put(session_id, cache)

        past_seq_len = cache.get_seq_length()
        position_ids = torch.arange(past_seq_len, past_seq_len + seq_len, device=device).unsqueeze(0)

        # Causal attention mask: only needed when input seq_len > 1 (e.g. prefill)
        causal_mask = None
        if seq_len > 1:
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=hidden_states.dtype),
                diagonal=1,
            )
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        # Compute rotary position embeddings
        position_embeddings = None
        if self.model_slice.rotary_emb is not None:
            position_embeddings = self.model_slice.rotary_emb(hidden_states, position_ids)

        # Run through each layer
        for layer in self.model_slice.layers:
            layer_output = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                past_key_values=cache,
                use_cache=True,
            )
            if isinstance(layer_output, tuple):
                hidden_states = layer_output[0]
            else:
                hidden_states = layer_output

        # If last node, apply final norm and LM head
        if self.is_last_node:
            if self.model_slice.norm is not None:
                hidden_states = self.model_slice.norm(hidden_states)
            if self.model_slice.lm_head is not None:
                hidden_states = self.model_slice.lm_head(hidden_states)

        return hidden_states

    async def serve_forever(self) -> None:
        """Run the node until cancelled."""
        await self.start()
        # Keep running
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await self.stop()

    async def stop(self) -> None:
        """Shut down the node."""
        await self.kv_store.stop_eviction_loop()
        self.kv_store.clear_all()
        if self._server:
            await self._server.stop()
        if self._next_client:
            await self._next_client.close()
        logger.info("Node stopped")


def main():
    """CLI entry point for starting a pipeline node."""
    parser = argparse.ArgumentParser(description="ShardFlow Pipeline Node")
    parser.add_argument("--model", required=True, help="Model path or HF model ID")
    parser.add_argument("--layer-start", type=int, required=True, help="First layer index (inclusive)")
    parser.add_argument("--layer-end", type=int, required=True, help="Last layer index (exclusive)")
    parser.add_argument("--host", default="0.0.0.0", help="Listen host")
    parser.add_argument("--port", type=int, default=9000, help="Listen port")
    parser.add_argument("--next-host", default=None, help="Next node host (omit for last node)")
    parser.add_argument("--next-port", type=int, default=None, help="Next node port")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--registry-url", default=None, help="Topology Registry URL for auto-registration")
    parser.add_argument("--public-host", default=None, help="Public address accessible by other nodes")
    parser.add_argument("--public-port", type=int, default=None, help="Public port accessible by other nodes")
    parser.add_argument("--node-id", default=None, help="Unique node identifier")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    is_last = args.next_host is None

    # Load model slice
    model_slice = load_layer_slice(
        model_path=args.model,
        layer_start=args.layer_start,
        layer_end=args.layer_end,
        include_norm=is_last,
        include_lm_head=is_last,
        device=args.device,
    )

    # Optional Auto-Registration with Topology Registry
    if args.registry_url:
        import requests, uuid
        node_id = args.node_id or f"node-{uuid.uuid4().hex[:6]}"
        pub_host = args.public_host or args.host
        pub_port = args.public_port or args.port
        vram = 0.0
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        try:
            resp = requests.post(
                f"{args.registry_url.rstrip('/')}/register",
                json={
                    "node_id": node_id,
                    "addr": pub_host,
                    "port": pub_port,
                    "layer_start": args.layer_start,
                    "layer_end": args.layer_end,
                    "vram_available_mb": vram,
                    "vram_total_mb": vram,
                },
                timeout=5.0,
            )
            if resp.status_code in (200, 201):
                logger.info("Registered node %s with registry at %s", node_id, args.registry_url)
        except Exception as e:
            logger.warning("Failed to register node with registry: %s", e)

    # Create and run node
    node = PipelineNode(
        model_slice=model_slice,
        is_first_node=(args.layer_start == 0),
        is_last_node=is_last,
        next_node_host=args.next_host,
        next_node_port=args.next_port,
        listen_host=args.host,
        listen_port=args.port,
    )

    asyncio.run(node.serve_forever())


if __name__ == "__main__":
    main()
