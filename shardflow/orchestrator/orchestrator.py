"""
Inference orchestrator — manages the decode loop, embedding, and sampling.

The orchestrator:
1. Loads the tokenizer and embedding layer (CPU — it's a lookup table)
2. Connects to Node 0 (sends activations) and the last node (receives logits)
3. Runs the per-token decode loop:
   - Embed token(s) → hidden_states
   - Send to Node 0 via TCP
   - Receive logits from last node
   - Sample next token
   - Repeat until EOS or max_tokens

Phase 1: Single request at a time, greedy decode, no KV cache (full sequence resent each iteration).
"""

import argparse
import asyncio
import logging
import time
import uuid
from typing import AsyncGenerator, Optional

import torch

from shardflow.orchestrator.tokenizer_utils import load_tokenizer
from shardflow.orchestrator.sampler import sample_next_token
from shardflow.transport.connection import NodeClient
from shardflow.transport.protocol import (
    MessageType,
    TensorMessage,
)

logger = logging.getLogger(__name__)


from shardflow.orchestrator.metrics import metrics


class PartialGenerationError(Exception):
    """Raised when node failure/disconnect occurs mid-generation, returning partial tokens."""

    def __init__(self, message: str, partial_text: str, tokens_generated: int, original_error: Exception):
        super().__init__(message)
        self.partial_text = partial_text
        self.tokens_generated = tokens_generated
        self.original_error = original_error


class Orchestrator:
    """
    Central controller for distributed inference.

    Owns: tokenizer, embedding layer, sampling logic, TCP connections to nodes.
    Does NOT own: transformer layers, KV cache.
    """

    def __init__(
        self,
        model_path: str,
        node_addresses: Optional[list[tuple[str, int]]] = None,
        registry_url: Optional[str] = None,
        device: str = "cpu",
        dtype: Optional[torch.dtype] = None,
        topology_ttl: float = 30.0,
    ):
        """
        Args:
            model_path: path to model (for tokenizer + embedding)
            node_addresses: optional explicit list of (host, port) for each node in order.
            registry_url: optional URL to FastAPI Topology Registry (e.g. http://registry:8001)
            device: device for embedding (default CPU — it's a lookup table)
            dtype: dtype for embedding weights
            topology_ttl: TTL in seconds for cached topology
        """
        self.model_path = model_path
        self.node_addresses = node_addresses or []
        self.registry_url = registry_url
        self.device = torch.device(device)
        self.dtype = dtype
        self.topology_ttl = topology_ttl

        # Topology caching
        self._cached_topology: list[tuple[str, int]] = self.node_addresses
        self._last_topology_fetch: float = 0.0

        # Loaded on init
        self.tokenizer = None
        self.embed_tokens = None
        self.config = None

        # TCP connection to Node 0 — logits flow back through the chain
        self._node0_client: Optional[NodeClient] = None

    def fetch_topology(self, force: bool = False) -> list[tuple[str, int]]:
        """Fetch topology from Registry URL if available, caching with TTL."""
        now = time.time()
        if not force and self._cached_topology and (now - self._last_topology_fetch < self.topology_ttl):
            return self._cached_topology

        # Primary path: direct in-memory lookup when registry is embedded in the gateway process.
        try:
            from shardflow.registry.app import get_topology
            topo_res = get_topology()
            if not topo_res.cluster_ready:
                logger.debug("In-memory cluster not ready yet (%d nodes)", topo_res.total_nodes)
            else:
                fetched = [(n.addr, n.port) for n in topo_res.nodes if n.is_active]
                if fetched:
                    self._cached_topology = fetched
                    self._last_topology_fetch = now
                    logger.info("Updated topology directly from in-memory registry: %s", fetched)
                    return self._cached_topology
        except Exception as e:
            logger.debug("In-memory topology fetch unavailable: %s", e)

        return self._cached_topology

    async def fetch_topology_async(self, force: bool = False) -> list[tuple[str, int]]:
        """Async topology fetch — avoids blocking the event loop on external registry HTTP."""
        now = time.time()
        if not force and self._cached_topology and (now - self._last_topology_fetch < self.topology_ttl):
            return self._cached_topology

        try:
            from shardflow.registry.app import get_topology
            topo_res = get_topology()
            if topo_res.cluster_ready:
                fetched = [(n.addr, n.port) for n in topo_res.nodes if n.is_active]
                if fetched:
                    self._cached_topology = fetched
                    self._last_topology_fetch = now
                    logger.info("Updated topology directly from in-memory registry: %s", fetched)
                    return self._cached_topology
        except Exception as e:
            logger.debug("In-memory topology fetch unavailable: %s", e)

        if self.registry_url:
            try:
                from shardflow.registry.client import async_get_topology
                fetched = await async_get_topology(self.registry_url)
                if fetched:
                    self._cached_topology = fetched
                    self._last_topology_fetch = now
                    logger.info("Updated topology from registry: %s", fetched)
                    return self._cached_topology
            except Exception as e:
                logger.warning("Failed to fetch topology from registry: %s", e)

        return self._cached_topology

    async def initialize(self) -> None:
        """Load tokenizer (zero model weights on CPU) and connect to Node 0."""
        logger.info("Initializing Zero-Weight Orchestrator for tokenizer %s...", self.model_path)
        self.tokenizer = load_tokenizer(self.model_path)

        # Connect to Node 0
        nodes = await self.fetch_topology_async(force=True)
        if not nodes:
            raise RuntimeError("No nodes available in topology")

        node0_host, node0_port = nodes[0]
        logger.info("Connecting orchestrator to Node 0 at %s:%d", node0_host, node0_port)
        self._node0_client = NodeClient(node0_host, node0_port)
        await self._node0_client.connect()

    def _embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Pass token IDs integer tensor directly (embedding executes on Node 0 GPU)."""
        return token_ids

    async def _ensure_node0_connected(self) -> None:
        """Ensure orchestrator is connected to current Node 0 target from topology."""
        nodes = await self.fetch_topology_async(force=True)
        if not nodes:
            raise RuntimeError("No active nodes available in topology")
        node0_host, node0_port = nodes[0]
        if (
            self._node0_client is None
            or not self._node0_client.is_connected
            or self._node0_client.host != node0_host
            or self._node0_client.port != node0_port
        ):
            if self._node0_client:
                try:
                    await self._node0_client.close()
                except Exception:
                    pass
            logger.info("Connecting orchestrator to Node 0 at %s:%d...", node0_host, node0_port)
            self._node0_client = NodeClient(node0_host, node0_port)
            await self._node0_client.connect()

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        stream: bool = False,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: input text
            max_tokens: max tokens to generate
            temperature: sampling temperature (0 = greedy)
            top_k: top-k sampling
            top_p: nucleus sampling
            stream: if True, print tokens as they're generated

        Returns:
            Generated text (completion only, not including prompt)
        """
        await self._ensure_node0_connected()
        session_id = str(uuid.uuid4())
        logger.info(
            "Starting generation: session=%s, max_tokens=%d, temp=%.2f",
            session_id, max_tokens, temperature,
        )

        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]  # [1, prompt_len]

        generated_tokens = []
        start_time = time.perf_counter()

        try:
            # Step 1: Chunked Prefill phase — chunk prompt into 512-token windows
            step_start = time.perf_counter()
            prompt_len = input_ids.shape[1]
            prefill_chunk_size = 512

            response = None
            for chunk_start in range(0, prompt_len, prefill_chunk_size):
                chunk_end = min(prompt_len, chunk_start + prefill_chunk_size)
                chunk_input_ids = input_ids[:, chunk_start:chunk_end]
                hidden_states = self._embed(chunk_input_ids)

                msg = TensorMessage(
                    msg_type=MessageType.ACTIVATION,
                    session_id=session_id,
                    tensor=hidden_states.cpu(),
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    sample_on_node=True,
                )
                response = await self._node0_client.send_recv(msg)

            if response is None:
                raise RuntimeError("No response from prefill phase")

            if response.msg_type == MessageType.TOKEN_ID:
                next_token = response.token_id
            elif response.msg_type == MessageType.LOGITS:
                logits = response.tensor[0, -1, :]
                next_token = sample_next_token(logits, temperature=temperature, top_k=top_k, top_p=top_p)
            else:
                raise RuntimeError(f"Expected TOKEN_ID or LOGITS from prefill, got {response.msg_type}")

            generated_tokens.append(next_token)

            token_text = self.tokenizer.decode([next_token])
            if stream:
                print(token_text, end="", flush=True)

            logger.debug(
                "Prefill step (len=%d): token=%d ('%s') in %.3fs",
                input_ids.shape[1], next_token, token_text.strip(), time.perf_counter() - step_start,
            )

            # Step 2: Incremental decode phase — send 1 token at a time
            for step in range(1, max_tokens):
                if next_token == self.tokenizer.eos_token_id:
                    logger.info("EOS reached at step %d", step)
                    break

                step_start = time.perf_counter()
                token_ids = torch.tensor([[next_token]], dtype=torch.long)
                hidden_states = self._embed(token_ids)  # [1, 1, hidden_dim]

                msg = TensorMessage(
                    msg_type=MessageType.ACTIVATION,
                    session_id=session_id,
                    tensor=hidden_states.cpu(),
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    sample_on_node=True,
                )
                response = await self._node0_client.send_recv(msg)

                if response.msg_type == MessageType.TOKEN_ID:
                    next_token = response.token_id
                elif response.msg_type == MessageType.LOGITS:
                    logits = response.tensor[0, -1, :]
                    next_token = sample_next_token(logits, temperature=temperature, top_k=top_k, top_p=top_p)
                else:
                    raise RuntimeError(f"Expected TOKEN_ID or LOGITS, got {response.msg_type}")

                generated_tokens.append(next_token)

                token_text = self.tokenizer.decode([next_token])
                if stream:
                    print(token_text, end="", flush=True)

                logger.debug(
                    "Decode step %d: token=%d ('%s') in %.3fs",
                    step, next_token, token_text.strip(), time.perf_counter() - step_start,
                )

        except Exception as err:
            partial = self.tokenizer.decode(generated_tokens, skip_special_tokens=True) if generated_tokens else ""
            logger.warning(
                "Generation interrupted after %d tokens due to node transport failure: %s",
                len(generated_tokens), err
            )
            raise PartialGenerationError(
                message=f"Node failure during generation step: {err}",
                partial_text=partial,
                tokens_generated=len(generated_tokens),
                original_error=err,
            )

        if stream:
            print()  # Newline after streaming

        total_time = time.perf_counter() - start_time
        num_tokens = len(generated_tokens)
        tok_per_sec = num_tokens / total_time if total_time > 0 else 0

        logger.info(
            "Generation complete: %d tokens in %.2fs (%.1f tok/s)",
            num_tokens, total_time, tok_per_sec,
        )

        # Send CLEAR to evict KV cache for this session
        clear_msg = TensorMessage(
            msg_type=MessageType.CLEAR,
            session_id=session_id,
            tensor=None,
        )
        try:
            await self._node0_client.send(clear_msg)
        except Exception as e:
            logger.debug("Failed to send CLEAR message: %s", e)

        # Decode the full completion
        completion = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return completion

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> AsyncGenerator[str, None]:
        """
        Async generator that yields decoded token strings one at a time.

        Handles prefill + decode loop. Sends CLEAR in finally so callers
        don't need to manage session cleanup on disconnect or error.
        """
        session_id = str(uuid.uuid4())
        await self._ensure_node0_connected()
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        prompt_len = input_ids.shape[1]

        try:
            # Prefill — chunked to avoid OOM on long prompts
            response = None
            for chunk_start in range(0, prompt_len, 512):
                chunk = input_ids[:, chunk_start:min(prompt_len, chunk_start + 512)]
                msg = TensorMessage(
                    msg_type=MessageType.ACTIVATION,
                    session_id=session_id,
                    tensor=chunk.cpu(),
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    sample_on_node=True,
                )
                response = await self._node0_client.send_recv(msg)

            if response is None:
                return

            if response.msg_type == MessageType.TOKEN_ID:
                next_token = response.token_id
            elif response.msg_type == MessageType.LOGITS:
                next_token = sample_next_token(
                    response.tensor[0, -1, :], temperature=temperature, top_k=top_k, top_p=top_p
                )
            else:
                raise RuntimeError(f"Unexpected prefill response: {response.msg_type}")

            yield self.tokenizer.decode([next_token])

            # Decode loop
            for _ in range(1, max_tokens):
                if next_token == self.tokenizer.eos_token_id:
                    break

                msg = TensorMessage(
                    msg_type=MessageType.ACTIVATION,
                    session_id=session_id,
                    tensor=torch.tensor([[next_token]], dtype=torch.long),
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    sample_on_node=True,
                )
                response = await self._node0_client.send_recv(msg)

                if response.msg_type == MessageType.TOKEN_ID:
                    next_token = response.token_id
                elif response.msg_type == MessageType.LOGITS:
                    next_token = sample_next_token(
                        response.tensor[0, -1, :], temperature=temperature, top_k=top_k, top_p=top_p
                    )
                else:
                    raise RuntimeError(f"Unexpected decode response: {response.msg_type}")

                yield self.tokenizer.decode([next_token])

        finally:
            # Always evict KV cache — fires on normal exit, error, or client disconnect
            clear_msg = TensorMessage(msg_type=MessageType.CLEAR, session_id=session_id, tensor=None)
            try:
                await self._node0_client.send(clear_msg)
            except Exception:
                pass

    async def shutdown(self) -> None:
        """Disconnect from all nodes."""
        if self._node0_client:
            await self._node0_client.close()
        logger.info("Orchestrator shutdown")


async def run_orchestrator(args):
    """Async entry point for the orchestrator CLI."""
    # Parse node addresses: "host1:port1,host2:port2"
    node_addresses = []
    for addr in args.nodes.split(","):
        host, port = addr.strip().split(":")
        node_addresses.append((host, int(port)))

    orchestrator = Orchestrator(
        model_path=args.model,
        node_addresses=node_addresses,
        device=args.device,
    )

    try:
        await orchestrator.initialize()

        result = await orchestrator.generate(
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            stream=True,
        )

        print(f"\n--- Full completion ---\n{result}")

    finally:
        await orchestrator.shutdown()


def main():
    """CLI entry point for the orchestrator."""
    parser = argparse.ArgumentParser(description="ShardFlow Orchestrator")
    parser.add_argument("--model", required=True, help="Model path or HF model ID")
    parser.add_argument(
        "--nodes", required=True,
        help="Comma-separated node addresses (host:port,host:port,...)"
    )
    parser.add_argument("--prompt", default="Hello, how are you?", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--device", default="cpu", help="Device for embedding (default: cpu)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    asyncio.run(run_orchestrator(args))


if __name__ == "__main__":
    main()
