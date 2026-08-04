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
from typing import Optional

import torch

from shardflow.node.layer_loader import load_layer_slice, load_tokenizer
from shardflow.orchestrator.sampler import sample_next_token
from shardflow.transport.connection import NodeClient
from shardflow.transport.protocol import (
    MessageType,
    TensorMessage,
)

logger = logging.getLogger(__name__)


from shardflow.orchestrator.metrics import metrics
import requests


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

        if self.registry_url:
            try:
                resp = requests.get(f"{self.registry_url.rstrip('/')}/topology", timeout=5.0)
                if resp.status_code == 200:
                    data = resp.json()
                    nodes = data.get("nodes", [])
                    fetched = [(n["addr"], n["port"]) for n in nodes]
                    if fetched:
                        self._cached_topology = fetched
                        self._last_topology_fetch = now
                        logger.info("Fetched topology from registry: %d nodes", len(fetched))
                        return self._cached_topology
            except Exception as e:
                logger.warning("Failed to fetch topology from registry (%s), using cached topology", e)

        return self._cached_topology

    async def initialize(self) -> None:
        """Load tokenizer + embedding, connect to nodes."""
        logger.info("Loading tokenizer from %s", self.model_path)
        self.tokenizer = load_tokenizer(self.model_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Loading embedding layer...")
        embed_slice = load_layer_slice(
            model_path=self.model_path,
            layer_start=0,
            layer_end=1,
            include_embed=True,
            dtype=self.dtype,
            device=str(self.device),
        )
        self.embed_tokens = embed_slice.embed_tokens
        self.config = embed_slice.config
        del embed_slice.layers
        logger.info(
            "Embedding loaded: vocab=%d, dim=%d, device=%s",
            self.config.vocab_size, self.config.hidden_size, self.device,
        )

        nodes = self.fetch_topology(force=True)
        if not nodes:
            raise RuntimeError("No active nodes available in topology!")

        first_host, first_port = nodes[0]
        self._node0_client = NodeClient(first_host, first_port, recv_timeout=120.0)
        await self._node0_client.connect()

        logger.info(
            "Connected to node chain (%d nodes). Ready for inference.",
            len(nodes),
        )

    @torch.inference_mode()
    def _embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Embed token IDs to hidden states.

        Args:
            token_ids: [1, seq_len] token IDs

        Returns:
            hidden_states: [1, seq_len, hidden_dim]
        """
        return self.embed_tokens(token_ids.to(self.device))

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

        Phase 1: No KV cache — resends the full growing sequence each iteration.
        This is O(n²) in sequence length. Phase 2 fixes this.

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
