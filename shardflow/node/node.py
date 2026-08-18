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
import time
import warnings
# ponytail: suppress transformers' max_cache_len DeprecationWarning — it fires from inside
# model layer forward passes we don't control (fixed upstream in transformers >= 5.16)
warnings.filterwarnings("ignore", message=".*max_cache_len.*deprecated.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*max_cache_len.*deprecated.*", category=FutureWarning)
from typing import Optional

import torch

from shardflow.node.layer_loader import load_layer_slice, ModelSlice
from shardflow.node.kv_cache import KVCacheStore
from shardflow.node.cuda_graph import CUDAGraphRunner
from shardflow.node.draft_model import DraftSampler, rewind_kv_cache
from shardflow.orchestrator.sampler import sample_next_token
from shardflow.transport.connection import NodeServer, NodeClient
from shardflow.transport.http_node import HTTPNodeClient, HTTPNodeServer
from shardflow.transport.protocol import (
    MessageType,
    TensorMessage,
)
from transformers.cache_utils import DynamicCache, StaticCache

logger = logging.getLogger(__name__)


class PipelineNode:
    """
    A single node in the inference pipeline.

    Holds a contiguous range of transformer layers and optionally
    the final norm + LM head (if this is the last node).
    Maintains a per-session KV cache with CUDA Graph replay acceleration.
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
        max_sessions: int = 4,
        enable_cuda_graphs: bool = True,
        draft_model: Optional[str] = None,
        draft_device: Optional[str] = None,
        spec_k: int = 4,
        next_node_url: Optional[str] = None,
        http_port: Optional[int] = None,
    ):
        self.model_slice = model_slice
        self.is_first_node = is_first_node
        self.is_last_node = is_last_node
        self.next_node_host = next_node_host
        self.next_node_port = next_node_port
        self.next_node_url = next_node_url
        self.http_port = http_port
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.enable_cuda_graphs = enable_cuda_graphs
        self.spec_k = spec_k

        # KV Cache Store (DynamicCache in eager mode, StaticCache in CUDA Graph mode)
        self.kv_store = KVCacheStore(
            eviction_timeout=kv_timeout,
            max_sessions=max_sessions,
            max_seq_len=2048,
            enable_static_cache=self.enable_cuda_graphs,
        )

        # Determine and cache node dtype safely (ponytail: cached once on init)
        self._node_dtype = torch.float16
        if model_slice.layers and len(model_slice.layers) > 0:
            params = list(model_slice.layers[0].parameters())
            if params and params[0].dtype in (torch.float16, torch.bfloat16, torch.float32):
                self._node_dtype = params[0].dtype

        hidden_size = 2048
        if model_slice.config is not None and hasattr(model_slice.config, "hidden_size"):
            hidden_size = model_slice.config.hidden_size
        elif model_slice.layers and len(model_slice.layers) > 0:
            for p in model_slice.layers[0].parameters():
                if len(p.shape) >= 2:
                    hidden_size = p.shape[-1]
                    break

        # CUDA Graph Runner
        self.graph_runner = CUDAGraphRunner(
            layers=model_slice.layers,
            hidden_size=hidden_size,
            device=model_slice.device,
            dtype=self._node_dtype,
            spec_k=spec_k,
            rotary_emb=model_slice.rotary_emb,
            enabled=enable_cuda_graphs,
        )

        # Speculative Decoding Draft Sampler on Node 0 (ponytail: lazy local draft generation)
        self.draft_sampler: Optional[DraftSampler] = None
        if self.is_first_node and draft_model and self.spec_k > 0:
            try:
                target_draft_device = torch.device(draft_device) if draft_device else model_slice.device
                self.draft_sampler = DraftSampler(
                    model_path=draft_model,
                    device=target_draft_device,
                    dtype=self._node_dtype,
                    spec_k=spec_k,
                    enable_cuda_graphs=enable_cuda_graphs,
                )
                logger.info("DraftSampler initialized on Node 0 (draft_model=%s, device=%s, K=%d)", draft_model, target_draft_device, spec_k)
                self.async_draft_sampler = None
            except Exception as e:
                logger.error("[ERROR] FAILED to initialize DraftSampler for '%s': %s", draft_model, e)
                print(f"\n[ERROR] [ERROR] Could not load draft model '{draft_model}': {e}\n", flush=True)
                self.draft_sampler = None
                self.async_draft_sampler = None

        # Connection to next node (if not the last)
        self._next_client: Optional[object] = None
        self._server: Optional[NodeServer] = None
        self._http_server: Optional[HTTPNodeServer] = None
        # Stream clients cache for terminal node stream-back: (host, port) -> NodeClient
        self._stream_clients: dict[tuple[str, int], NodeClient] = {}
        # Active session tasks for immediate cancellation on disconnect/new session: session_id -> (Task, cancelled_flag)
        self._active_session_tasks: dict[str, tuple[asyncio.Task, list[bool]]] = {}

    async def start(self) -> None:
        """Start the node: connect to next node, start listening, start eviction loop, capture CUDA graphs."""
        node_dtype = torch.float16
        if self.model_slice.layers and len(self.model_slice.layers) > 0:
            params = list(self.model_slice.layers[0].parameters())
            if params:
                node_dtype = params[0].dtype

        if self.model_slice.config is not None:
            self.kv_store.initialize_static_pool(
                config=self.model_slice.config,
                device=self.model_slice.device,
                dtype=node_dtype,
            )
            if self.kv_store._static_slots and self.enable_cuda_graphs:
                self.graph_runner.capture(self.kv_store._static_slots[0].cache)

        await self.kv_store.start_eviction_loop()

        # 1. Start TCP server immediately so this node is listening for incoming connections
        self._server = NodeServer(
            host=self.listen_host,
            port=self.listen_port,
            handler=self._handle_message,
        )
        await self._server.start()

        # 1b. Start HTTP server if http_port is configured (for remote tunnel/WAN transport)
        if self.http_port is not None:
            self._http_server = HTTPNodeServer(
                host=self.listen_host,
                port=self.http_port,
                handler=self._handle_message,
            )
            await self._http_server.start()

        # 2. Connect to the next node in the chain (if not last)
        if not self.is_last_node:
            if self.next_node_url:
                self._next_client = HTTPNodeClient(self.next_node_url)
                logger.info("Configured HTTP client to next node at %s", self.next_node_url)
            elif self.next_node_host:
                self._next_client = NodeClient(
                    self.next_node_host,
                    self.next_node_port,
                )
                try:
                    await self._next_client.connect(max_retries=5, retry_delay=1.0)
                    logger.info(
                        "Connected to next node at %s:%d",
                        self.next_node_host, self.next_node_port
                    )
                except Exception as e:
                    logger.info(
                        "Next node at %s:%d is still booting — will connect on first request: %s",
                        self.next_node_host, self.next_node_port, e
                    )

        logger.info(
            "Node ready — layers [%d, %d), %s, listening on %s:%d (CUDA Graphs: %s)",
            self.model_slice.layer_start,
            self.model_slice.layer_end,
            "LAST node (has LM head)" if self.is_last_node else "INTERMEDIATE node",
            self.listen_host,
            self.listen_port,
            self.graph_runner.is_captured,
        )

    async def update_next_node(self, host: Optional[str], port: Optional[int]) -> None:
        """
        Dynamically update next-node routing target.
        Handles None (disconnect/eviction), unchanged targets (no-op), and new targets (reconnect).
        """
        if self.next_node_url and self._next_client is not None:
            return

        if host == self.next_node_host and port == self.next_node_port and self._next_client is not None and getattr(self._next_client, "is_connected", False):
            return

        # Close old client if active
        if self._next_client is not None:
            logger.info("Closing old connection to next node (%s:%s)...", self.next_node_host, self.next_node_port)
            try:
                await self._next_client.close()
            except Exception as e:
                logger.debug("Error closing old next-node client: %s", e)
            self._next_client = None

        self.next_node_host = host
        self.next_node_port = port

        if not self.is_last_node and host and port:
            logger.info("Connecting to updated next node target at %s:%d...", host, port)
            self._next_client = NodeClient(host, port)
            try:
                await self._next_client.connect()
                logger.info("Successfully connected to next node at %s:%d", host, port)
            except Exception as e:
                logger.warning("Could not connect to updated next node %s:%d: %s", host, port, e)
                self._next_client = None

    async def _get_stream_client(self, host: str, port: int) -> Optional[NodeClient]:
        """Get or create a cached stream-back client to the Gateway."""
        if not host or not port:
            return None

        key = (host, port)
        client = self._stream_clients.get(key)
        if client is not None and client.is_connected:
            return client

        try:
            client = NodeClient(host, port, send_timeout=5.0, recv_timeout=10.0)
            await client.connect(max_retries=2, retry_delay=0.2)
            self._stream_clients[key] = client
            return client
        except Exception as e:
            logger.debug("Could not establish stream-back connection to %s:%d: %s", host, port, e)
            return None

    def _get_cache_seq_len(self, cache) -> int:
        """Get sequence length for this node's layer slice cache."""
        if cache is None:
            return 0
        if hasattr(cache, "_seen_tokens") and cache._seen_tokens is not None:
            return int(cache._seen_tokens)
        if hasattr(cache, "get_seq_length"):
            try:
                seq = cache.get_seq_length(self.model_slice.layer_start)
                if isinstance(seq, torch.Tensor):
                    return int(seq.item())
                if seq is not None and int(seq) >= 0:
                    return int(seq)
            except Exception:
                pass
            try:
                seq = cache.get_seq_length()
                if isinstance(seq, torch.Tensor):
                    return int(seq.item())
                if seq is not None and int(seq) >= 0:
                    return int(seq)
            except Exception:
                pass
            try:
                seq = cache.get_seq_length(0)
                if isinstance(seq, torch.Tensor):
                    return int(seq.item())
                if seq is not None and int(seq) >= 0:
                    return int(seq)
            except Exception:
                pass
        if hasattr(cache, "key_cache") and len(cache.key_cache) > 0:
            try:
                return int(cache.key_cache[0].shape[-2])
            except Exception:
                pass
        return 0

    async def _handle_message(self, msg: TensorMessage) -> Optional[TensorMessage]:
        """
        Process an incoming message.

        For CLEAR: evict KV cache for session_id.
        For START_SESSION (v2 Data Plane): Node 0 drives the full decode loop across the cluster.
        For ACTIVATION:
        - Run hidden states through local layers using cached KV
        - If last node: sample token ID, stream back to Gateway if stream_back address is set, return token_id
        - If not last: forward to next node, wait for response, pass back
        """
        if msg.msg_type == MessageType.CLEAR:
            if msg.session_id in self._active_session_tasks:
                task, flag = self._active_session_tasks.pop(msg.session_id)
                flag[0] = True
                task.cancel()
                logger.info("Cancelled active session task %s on CLEAR", msg.session_id)
            self.kv_store.evict(msg.session_id)
            if not self.is_last_node and self._next_client and getattr(self._next_client, "is_connected", False):
                try:
                    await self._next_client.send(msg)
                except (ConnectionError, OSError):
                    logger.debug("Could not forward CLEAR — next node already disconnected")
            return None

        if msg.msg_type == MessageType.START_SESSION:
            # Cancel any prior running generation tasks immediately via flag + task.cancel()
            for active_sid, (active_task, active_flag) in list(self._active_session_tasks.items()):
                active_flag[0] = True
                active_task.cancel()
                self.kv_store.evict(active_sid)
                self._active_session_tasks.pop(active_sid, None)
                logger.info("Cancelled prior running session %s for incoming session %s", active_sid, msg.session_id)

            cancelled_flag = [False]
            task = asyncio.ensure_future(self._handle_start_session(msg, cancelled_flag))
            self._active_session_tasks[msg.session_id] = (task, cancelled_flag)
            try:
                return await task
            except asyncio.CancelledError:
                return TensorMessage(msg_type=MessageType.TOKEN_ID, session_id=msg.session_id, token_id=0, is_eos=True)
            finally:
                self._active_session_tasks.pop(msg.session_id, None)

        if msg.msg_type != MessageType.ACTIVATION:
            logger.warning("Unknown message type: %s", msg.msg_type)
            return None

        # Process activation / token IDs
        tensor = msg.tensor.to(self.model_slice.device, non_blocking=False)
        if self.is_first_node and tensor.dtype in (torch.long, torch.int64, torch.int32):
            if self.model_slice.embed_tokens is not None:
                hidden_states = self.model_slice.embed_tokens(tensor)
            else:
                hidden_states = tensor
        else:
            hidden_states = tensor

        output = self._forward(
            hidden_states,
            session_id=msg.session_id,
            compute_head=msg.sample_on_node if self.is_last_node else False,
        )

        if self.is_last_node:
            if msg.sample_on_node:
                if msg.draft_tokens:
                    # ponytail: causal speculative candidate verification on terminal node
                    # Input contains [T_current, d_0, ..., d_{K-1}] (length K+1)
                    # Output at index i predicts token following candidate_tokens[i] -> compare with drafts[i]
                    drafts = msg.draft_tokens
                    accepted_tokens = []
                    next_token = None
                    for i in range(len(drafts)):
                        cand = sample_next_token(
                            output[0, i, :],
                            temperature=msg.temperature,
                            top_k=msg.top_k,
                            top_p=msg.top_p,
                        )
                        if cand == drafts[i]:
                            accepted_tokens.append(drafts[i])
                        else:
                            next_token = cand
                            break

                    if next_token is None:
                        # All K drafts verified; bonus sample from final candidate position (K)
                        next_token = sample_next_token(
                            output[0, -1, :],
                            temperature=msg.temperature,
                            top_k=msg.top_k,
                            top_p=msg.top_p,
                        )

                    accepted_count = len(accepted_tokens) + 1

                    # Rewind terminal KV cache to exact accepted sequence length
                    cache = self.kv_store.get(msg.session_id)
                    if cache is not None:
                        past_seq = self._get_cache_seq_len(cache)
                        past_seq_before = int(past_seq or 0) - (len(drafts) + 1)
                        rewind_kv_cache(cache, past_seq_before + accepted_count)

                    # Stream accepted tokens directly to Gateway
                    if msg.stream_back_host and msg.stream_back_port:
                        try:
                            stream_client = await self._get_stream_client(msg.stream_back_host, msg.stream_back_port)
                            if stream_client and stream_client.is_connected:
                                for tok in accepted_tokens:
                                    await stream_client.send(TensorMessage(
                                        msg_type=MessageType.STREAM_TOKEN,
                                        session_id=msg.session_id,
                                        token_id=tok,
                                        is_eos=False,
                                    ))
                                await stream_client.send(TensorMessage(
                                    msg_type=MessageType.STREAM_TOKEN,
                                    session_id=msg.session_id,
                                    token_id=next_token,
                                    is_eos=False,
                                ))
                        except Exception as e:
                            logger.debug("Stream-back failed: %s", e)

                    return TensorMessage(
                        msg_type=MessageType.TOKEN_ID,
                        session_id=msg.session_id,
                        token_id=next_token,
                        accepted_count=accepted_count,
                    )
                else:
                    logits = output[0, -1, :]
                    token_id = sample_next_token(
                        logits,
                        temperature=msg.temperature,
                        top_k=msg.top_k,
                        top_p=msg.top_p,
                    )

                    # Direct stream-back to Gateway over dedicated TCP channel
                    if msg.stream_back_host and msg.stream_back_port:
                        try:
                            stream_client = await self._get_stream_client(msg.stream_back_host, msg.stream_back_port)
                            if stream_client and stream_client.is_connected:
                                stream_msg = TensorMessage(
                                    msg_type=MessageType.STREAM_TOKEN,
                                    session_id=msg.session_id,
                                    token_id=token_id,
                                    is_eos=False,
                                )
                                await stream_client.send(stream_msg)
                        except Exception as e:
                            logger.debug("Stream-back to %s:%d failed: %s", msg.stream_back_host, msg.stream_back_port, e)

                    return TensorMessage(
                        msg_type=MessageType.TOKEN_ID,
                        session_id=msg.session_id,
                        token_id=token_id,
                    )
            else:
                return TensorMessage(
                    msg_type=MessageType.ACTIVATION,
                    session_id=msg.session_id,
                    tensor=output.to("cpu", non_blocking=True),
                    temperature=msg.temperature,
                    top_k=msg.top_k,
                    top_p=msg.top_p,
                    sample_on_node=False,
                    draft_tokens=msg.draft_tokens,
                )
        else:
            if self._next_client is None or not self._next_client.is_connected:
                if self.next_node_host and self.next_node_port:
                    logger.warning("Next client disconnected — attempting emergency reconnect to %s:%d...", self.next_node_host, self.next_node_port)
                    await self.update_next_node(self.next_node_host, self.next_node_port)

            if self._next_client is None and self.next_node_host and self.next_node_port:
                self._next_client = NodeClient(self.next_node_host, self.next_node_port)

            if self._next_client is None:
                logger.error("Cannot forward activation — next node target is not configured")
                raise RuntimeError(f"Next node target is not configured")

            forward_msg = TensorMessage(
                msg_type=MessageType.ACTIVATION,
                session_id=msg.session_id,
                tensor=output.to("cpu", non_blocking=True),
                temperature=msg.temperature,
                top_k=msg.top_k,
                top_p=msg.top_p,
                sample_on_node=msg.sample_on_node,
                stream_back_host=msg.stream_back_host,
                stream_back_port=msg.stream_back_port,
                draft_tokens=msg.draft_tokens,
            )
            try:
                response = await self._next_client.send_recv(forward_msg, timeout=20.0)
                # Rewind intermediate node KV cache on speculative verification mismatch
                if msg.draft_tokens and response.msg_type == MessageType.TOKEN_ID:
                    cache = self.kv_store.get(msg.session_id)
                    if cache is not None:
                        past_seq = self._get_cache_seq_len(cache)
                        past_seq_before = int(past_seq or 0) - (len(msg.draft_tokens) + 1)
                        accepted_count = response.accepted_count or 1
                        rewind_kv_cache(cache, past_seq_before + accepted_count)
                return response
            except (asyncio.TimeoutError, ConnectionError, OSError, Exception) as err:
                logger.error(
                    "Forward to next node (%s:%s) failed or timed out for session %s: %s",
                    self.next_node_host, self.next_node_port, msg.session_id, err,
                )
                raise RuntimeError(
                    f"Forward to next node ({self.next_node_host}:{self.next_node_port}) failed: {err}"
                ) from err

    async def _handle_start_session(self, msg: TensorMessage, cancelled_flag: Optional[list[bool]] = None) -> TensorMessage:
        """
        v2 Data-Plane Controller on Node 0.

        Drives the complete autoregressive generation loop peer-to-peer across
        worker GPU nodes without involving the Gateway on every token.
        """
        session_id = msg.session_id
        prompt_tokens = msg.prompt_tokens or []
        max_tokens = msg.max_tokens or 100
        temperature = msg.temperature
        top_k = msg.top_k
        top_p = msg.top_p
        eos_id = msg.eos_token_id
        stream_host = msg.stream_back_host
        stream_port = msg.stream_back_port

        # When using HTTP transport, Node 0 streams locally to Gateway; do not instruct Node 1 to stream back over TCP
        next_stream_host = None if isinstance(self._next_client, HTTPNodeClient) else stream_host
        next_stream_port = None if isinstance(self._next_client, HTTPNodeClient) else stream_port

        logger.info(
            "Starting v2 data-plane generation: session=%s, prompt_len=%d, max_tokens=%d (speculative=%s)",
            session_id, len(prompt_tokens), max_tokens, (self.draft_sampler is not None and self.spec_k > 0),
        )

        try:
            # Prefill draft sampler with prompt context for aligned speculative proposals
            if self.draft_sampler is not None and self.spec_k > 0:
                self.draft_sampler.prefill(prompt_tokens)

            # 1. Chunked Prefill Phase (windows of 512 tokens)
            prefill_chunk_size = 512
            prompt_len = len(prompt_tokens)
            next_token = None

            for chunk_start in range(0, prompt_len, prefill_chunk_size):
                chunk_end = min(prompt_len, chunk_start + prefill_chunk_size)
                chunk = prompt_tokens[chunk_start:chunk_end]
                is_final_chunk = (chunk_end == prompt_len)

                token_tensor = torch.tensor([chunk], dtype=torch.long, device=self.model_slice.device)
                if self.model_slice.embed_tokens is not None:
                    hidden_states = self.model_slice.embed_tokens(token_tensor)
                else:
                    hidden_states = token_tensor

                output = self._forward(
                    hidden_states,
                    session_id=session_id,
                    compute_head=is_final_chunk if self.is_last_node else False,
                )

                if self.is_last_node:
                    if is_final_chunk:
                        logits = output[0, -1, :]
                        next_token = sample_next_token(logits, temperature=temperature, top_k=top_k, top_p=top_p)
                        if stream_host and stream_port:
                            stream_client = await self._get_stream_client(stream_host, stream_port)
                            if stream_client and stream_client.is_connected:
                                await stream_client.send(TensorMessage(
                                    msg_type=MessageType.STREAM_TOKEN,
                                    session_id=session_id,
                                    token_id=next_token,
                                    is_eos=False,
                                ))
                else:
                    if self._next_client is None or not self._next_client.is_connected:
                        await self.update_next_node(self.next_node_host, self.next_node_port)
                    if output.is_cuda:
                        torch.cuda.synchronize(output.device)
                    forward_msg = TensorMessage(
                        msg_type=MessageType.ACTIVATION,
                        session_id=session_id,
                        tensor=output.to("cpu", non_blocking=False),
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        sample_on_node=is_final_chunk,
                        stream_back_host=next_stream_host,
                        stream_back_port=next_stream_port,
                    )
                    resp = await self._next_client.send_recv(forward_msg, timeout=30.0)
                    if is_final_chunk and resp.msg_type == MessageType.TOKEN_ID:
                        next_token = resp.token_id
                        # HTTP transport: stream prefill token from Node 0 locally to Gateway
                        if isinstance(self._next_client, HTTPNodeClient) and stream_host and stream_port:
                            stream_client = await self._get_stream_client(stream_host, stream_port)
                            if stream_client and stream_client.is_connected:
                                await stream_client.send(TensorMessage(
                                    msg_type=MessageType.STREAM_TOKEN,
                                    session_id=session_id,
                                    token_id=next_token,
                                    is_eos=False,
                                ))

            if next_token is None:
                raise RuntimeError("No token returned from prefill phase")

            # 2. Peer-to-Peer Decode Loop (with speculative acceleration when DraftSampler is present)
            step = 1
            while step < max_tokens:
                # Direct cancellation check via mutable flag — guaranteed to abort immediately
                if cancelled_flag and cancelled_flag[0]:
                    logger.info("Session %s aborted via cancelled_flag", session_id)
                    break

                if eos_id is not None and next_token == eos_id:
                    logger.info("Session %s reached EOS at step %d", session_id, step)
                    break

                if self.draft_sampler is not None and self.spec_k > 0:
                    # Speculative decode: generate K draft tokens locally on Node 0 GPU
                    drafts = self.draft_sampler.generate_drafts(
                        next_token,
                        k=self.spec_k,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                    )
                    # Candidate sequence: [T_current, d_0, ..., d_{K-1}] (length K+1)
                    candidate_tokens = [next_token] + drafts
                    cand_tensor = torch.tensor([candidate_tokens], dtype=torch.long, device=self.model_slice.device)
                    if self.model_slice.embed_tokens is not None:
                        hidden_states = self.model_slice.embed_tokens(cand_tensor)
                    else:
                        hidden_states = cand_tensor

                    cache = self.kv_store.get(session_id)
                    past_seq_len = self._get_cache_seq_len(cache)

                    output = self._forward(
                        hidden_states,
                        session_id=session_id,
                        compute_head=self.is_last_node,
                    )

                    if self.is_last_node:
                        # Single-node mode with speculative decoding
                        accepted_tokens = []
                        corrected = None
                        for i in range(len(drafts)):
                            cand = sample_next_token(output[0, i, :], temperature=temperature, top_k=top_k, top_p=top_p)
                            if cand == drafts[i]:
                                accepted_tokens.append(drafts[i])
                            else:
                                corrected = cand
                                break
                        next_token = corrected if corrected is not None else sample_next_token(output[0, -1, :], temperature=temperature, top_k=top_k, top_p=top_p)
                        accepted_count = len(accepted_tokens) + 1
                        if cache is not None:
                            rewind_kv_cache(cache, past_seq_len + accepted_count)
                        if self.draft_sampler:
                            # Rewind draft model to its own position (independent of target model)
                            draft_target = self.draft_sampler.seq_len - len(drafts) + accepted_count
                            self.draft_sampler.rewind(draft_target)

                        if stream_host and stream_port:
                            stream_client = await self._get_stream_client(stream_host, stream_port)
                            if stream_client and stream_client.is_connected:
                                for tok in accepted_tokens:
                                    await stream_client.send(TensorMessage(
                                        msg_type=MessageType.STREAM_TOKEN,
                                        session_id=session_id,
                                        token_id=tok,
                                        is_eos=False,
                                    ))
                                await stream_client.send(TensorMessage(
                                    msg_type=MessageType.STREAM_TOKEN,
                                    session_id=session_id,
                                    token_id=next_token,
                                    is_eos=False,
                                ))
                        step += accepted_count
                    else:
                        if output.is_cuda:
                            torch.cuda.synchronize(output.device)
                        forward_msg = TensorMessage(
                            msg_type=MessageType.ACTIVATION,
                            session_id=session_id,
                            tensor=output.to("cpu", non_blocking=False),
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                            sample_on_node=True,
                            stream_back_host=next_stream_host,
                            stream_back_port=next_stream_port,
                            draft_tokens=drafts,
                        )
                        resp = await self._next_client.send_recv(forward_msg, timeout=20.0)
                        if resp.msg_type == MessageType.TOKEN_ID:
                            next_token = resp.token_id
                            accepted_count = resp.accepted_count or 1
                            if cache is not None:
                                rewind_kv_cache(cache, past_seq_len + accepted_count)
                            if self.draft_sampler:
                                # Rewind draft model using its own seq_len, not the target model's
                                draft_target = self.draft_sampler.seq_len - len(drafts) + accepted_count
                                self.draft_sampler.rewind(draft_target)

                            # HTTP transport: stream accepted tokens from Node 0 locally to Gateway
                            if isinstance(self._next_client, HTTPNodeClient) and stream_host and stream_port:
                                stream_client = await self._get_stream_client(stream_host, stream_port)
                                if stream_client and stream_client.is_connected:
                                    if drafts and accepted_count > 1:
                                        for d_idx in range(accepted_count - 1):
                                            await stream_client.send(TensorMessage(
                                                msg_type=MessageType.STREAM_TOKEN,
                                                session_id=session_id,
                                                token_id=drafts[d_idx],
                                                is_eos=False,
                                            ))
                                    await stream_client.send(TensorMessage(
                                        msg_type=MessageType.STREAM_TOKEN,
                                        session_id=session_id,
                                        token_id=next_token,
                                        is_eos=False,
                                    ))

                            step += accepted_count
                        else:
                            raise RuntimeError(f"Unexpected response in speculative decode loop: {resp.msg_type}")
                else:
                    # Standard 1-token decode loop (lock-free & non-blocking)
                    token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=self.model_slice.device)
                    if self.model_slice.embed_tokens is not None:
                        hidden_states = self.model_slice.embed_tokens(token_tensor)
                    else:
                        hidden_states = token_tensor

                    output = self._forward(
                        hidden_states,
                        session_id=session_id,
                        compute_head=self.is_last_node,
                    )

                    if self.is_last_node:
                        logits = output[0, -1, :]
                        next_token = sample_next_token(logits, temperature=temperature, top_k=top_k, top_p=top_p)
                        if stream_host and stream_port:
                            stream_client = await self._get_stream_client(stream_host, stream_port)
                            if stream_client and stream_client.is_connected:
                                await stream_client.send(TensorMessage(
                                    msg_type=MessageType.STREAM_TOKEN,
                                    session_id=session_id,
                                    token_id=next_token,
                                    is_eos=False,
                                ))
                    else:
                        if output.is_cuda:
                            torch.cuda.synchronize(output.device)
                        forward_msg = TensorMessage(
                            msg_type=MessageType.ACTIVATION,
                            session_id=session_id,
                            tensor=output.to("cpu", non_blocking=False),
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                            sample_on_node=True,
                            stream_back_host=next_stream_host,
                            stream_back_port=next_stream_port,
                        )
                        resp = await self._next_client.send_recv(forward_msg, timeout=15.0)
                        if resp.msg_type == MessageType.TOKEN_ID:
                            next_token = resp.token_id
                            # HTTP transport: stream token from Node 0 locally to Gateway
                            if isinstance(self._next_client, HTTPNodeClient) and stream_host and stream_port:
                                stream_client = await self._get_stream_client(stream_host, stream_port)
                                if stream_client and stream_client.is_connected:
                                    await stream_client.send(TensorMessage(
                                        msg_type=MessageType.STREAM_TOKEN,
                                        session_id=session_id,
                                        token_id=next_token,
                                        is_eos=False,
                                    ))
                        else:
                            raise RuntimeError(f"Unexpected response in decode loop: {resp.msg_type}")
                    step += 1

        finally:
            # Cleanup: evict KV cache across cluster
            self.kv_store.evict(session_id)
            if not self.is_last_node and self._next_client and self._next_client.is_connected:
                try:
                    await self._next_client.send(TensorMessage(
                        msg_type=MessageType.CLEAR,
                        session_id=session_id,
                    ))
                except Exception:
                    pass

            # Signal stream completion to Gateway
            if stream_host and stream_port:
                try:
                    stream_client = await self._get_stream_client(stream_host, stream_port)
                    if stream_client and stream_client.is_connected:
                        await stream_client.send(TensorMessage(
                            msg_type=MessageType.STREAM_TOKEN,
                            session_id=session_id,
                            token_id=0,
                            is_eos=True,
                            finish_reason="stop",
                        ))
                except Exception:
                    pass

        return TensorMessage(
            msg_type=MessageType.TOKEN_ID,
            session_id=session_id,
            token_id=next_token,
        )

    @torch.inference_mode()
    def _forward(
        self,
        hidden_states: torch.Tensor,
        session_id: str,
        compute_head: bool = True,
    ) -> torch.Tensor:
        """
        Run hidden states through this node's layers with KV caching and CUDA Graph replay.

        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            session_id: unique session ID for KV cache lookup
            compute_head: if True and last node, apply final norm and LM head

        Returns:
            If last node and compute_head: logits [batch, seq_len, vocab_size]
            Otherwise: hidden_states [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device

        # Lookup or lease static KV cache slot (or dynamic fallback)
        cache = self.kv_store.get_or_create(
            session_id,
            config=self.model_slice.config,
            device=self.model_slice.device,
            dtype=getattr(self, "_node_dtype", torch.float16),
        )

        past_seq_len = self._get_cache_seq_len(cache)

        # Fast path: CUDA Graph replay for single-token autoregressive decoding and speculative verify
        if seq_len == 1 and self.graph_runner.can_use_graph(seq_len):
            hidden_states = self.graph_runner.replay_decode(hidden_states, past_seq_len)
            # ponytail: CUDA Graph only replays GPU kernels — Python-side _seen_tokens is NOT
            # updated by graph.replay(). Manually advance it so the next get_seq_length() call
            # returns the correct position. Without this, every decode step reads past_seq_len=N
            # (frozen at prefill end) and all KV writes land on the same slot → gibberish.
            if cache is not None and hasattr(cache, "_seen_tokens"):
                cache._seen_tokens = past_seq_len + 1
        elif seq_len == (self.graph_runner.spec_k + 1) and self.graph_runner.can_use_graph(seq_len):
            # ponytail: fast CUDA Graph replay for speculative verification of K candidate tokens
            hidden_states = self.graph_runner.replay_verify(hidden_states, past_seq_len)
            if cache is not None and hasattr(cache, "_seen_tokens"):
                cache._seen_tokens = past_seq_len + seq_len
        else:
            # Eager execution for prefill / multi-token sequences
            position_ids = torch.arange(past_seq_len, past_seq_len + seq_len, device=device).unsqueeze(0)

            # Causal attention mask: ensure SDPA never attends to unwritten trailing slots in StaticCache
            causal_mask = None
            if isinstance(cache, StaticCache):
                try:
                    max_len = cache.get_max_cache_shape() if hasattr(cache, "get_max_cache_shape") else getattr(cache, "max_cache_len", None)
                    if max_len is not None:
                        key_len = int(max_len)
                        causal_mask = torch.full(
                            (1, 1, seq_len, key_len),
                            float("-inf"),
                            device=device,
                            dtype=hidden_states.dtype,
                        )
                        for i in range(seq_len):
                            causal_mask[0, 0, i, : past_seq_len + i + 1] = 0.0
                except Exception as e:
                    logger.debug("StaticCache causal mask construction fallback: %s", e)
            elif seq_len > 1:
                key_len = past_seq_len + seq_len
                causal_mask = torch.full(
                    (seq_len, key_len),
                    0.0,
                    device=device,
                    dtype=hidden_states.dtype,
                )
                current_chunk_mask = torch.triu(
                    torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=hidden_states.dtype),
                    diagonal=1,
                )
                causal_mask[:, past_seq_len : past_seq_len + seq_len] = current_chunk_mask
                causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

            # Compute rotary position embeddings
            position_embeddings = None
            if self.model_slice.rotary_emb is not None:
                try:
                    position_embeddings = self.model_slice.rotary_emb(hidden_states, position_ids)
                except Exception as e:
                    logger.debug("rotary_emb compute fallback: %s", e)

            # Check layer signature once to avoid inner-loop TypeError exception overhead
            if not hasattr(self, "_layer_accepts_pos_emb"):
                self._layer_accepts_pos_emb = False
                if self.model_slice.layers and len(self.model_slice.layers) > 0:
                    import inspect
                    try:
                        sig = inspect.signature(self.model_slice.layers[0].forward)
                        self._layer_accepts_pos_emb = "position_embeddings" in sig.parameters
                    except Exception:
                        self._layer_accepts_pos_emb = False

            cache_position = position_ids.squeeze(0)

            # Detailed timing breakdown for multi-GPU forward
            dev0_time = 0.0
            pcie_time = 0.0
            dev1_time = 0.0
            t_sub0 = time.perf_counter()

            # Run through each layer with direct kwargs
            for layer in self.model_slice.layers:
                layer_params = list(layer.parameters())
                if layer_params:
                    layer_dev = layer_params[0].device
                    if hidden_states.device != layer_dev:
                        if hidden_states.is_cuda:
                            torch.cuda.synchronize(hidden_states.device)
                        t_sub1 = time.perf_counter()
                        dev0_time += (t_sub1 - t_sub0) * 1000.0

                        t_pcie0 = time.perf_counter()
                        hidden_states = hidden_states.to(layer_dev, non_blocking=False)
                        position_ids = position_ids.to(layer_dev, non_blocking=False)
                        cache_position = cache_position.to(layer_dev, non_blocking=False)
                        if causal_mask is not None:
                            causal_mask = causal_mask.to(layer_dev, non_blocking=False)
                        if position_embeddings is not None and isinstance(position_embeddings, (tuple, list)):
                            position_embeddings = (
                                position_embeddings[0].to(layer_dev),
                                position_embeddings[1].to(layer_dev),
                            )
                        if hidden_states.is_cuda:
                            torch.cuda.synchronize(hidden_states.device)
                        t_pcie1 = time.perf_counter()
                        pcie_time += (t_pcie1 - t_pcie0) * 1000.0
                        t_sub0 = time.perf_counter()

                kwargs = {
                    "attention_mask": causal_mask,
                    "position_ids": position_ids,
                    "past_key_values": cache,
                    "use_cache": True,
                    "cache_position": cache_position,
                }
                if self._layer_accepts_pos_emb and position_embeddings is not None:
                    kwargs["position_embeddings"] = position_embeddings

                layer_output = layer(hidden_states, **kwargs)

                if isinstance(layer_output, tuple):
                    hidden_states = layer_output[0]
                else:
                    hidden_states = layer_output

            if hidden_states.is_cuda:
                torch.cuda.synchronize(hidden_states.device)
            t_sub_end = time.perf_counter()
            dev1_time += (t_sub_end - t_sub0) * 1000.0

            self.last_forward_breakdown = {
                "gpu0_ms": dev0_time if pcie_time > 0 else dev1_time,
                "pcie_ms": pcie_time,
                "gpu1_ms": dev1_time if pcie_time > 0 else 0.0,
                "total_fwd_ms": dev0_time + pcie_time + dev1_time,
            }

            # Eager path: StaticCache._seen_tokens is not updated by layer.forward();
            # manually advance it so get_seq_length() returns the correct next position
            # on the next decode step. Without this, every step reads past_seq_len=N
            # (frozen at prefill end) and all KV writes land on the same slot → gibberish.
            if cache is not None and hasattr(cache, "_seen_tokens"):
                cache._seen_tokens = past_seq_len + seq_len

        # If last node and compute_head requested, apply final norm and LM head
        if self.is_last_node and compute_head:
            if self.model_slice.norm is not None:
                norm_dev = next(self.model_slice.norm.parameters()).device
                if hidden_states.device != norm_dev:
                    hidden_states = hidden_states.to(norm_dev, non_blocking=False)
                hidden_states = self.model_slice.norm(hidden_states)
            if self.model_slice.lm_head is not None:
                head_dev = next(self.model_slice.lm_head.parameters()).device
                if hidden_states.device != head_dev:
                    hidden_states = hidden_states.to(head_dev, non_blocking=False)
                hidden_states = self.model_slice.lm_head(hidden_states)

        return hidden_states

    async def serve_forever(self) -> None:
        """Run the node until cancelled, logging periodic health telemetry every 10 seconds."""
        await self.start()
        try:
            while True:
                await asyncio.sleep(10.0)
                vram_str = "N/A"
                if self.model_slice.device.type == "cuda" and torch.cuda.is_available():
                    alloc_gb = torch.cuda.memory_allocated(self.model_slice.device) / 1024**3
                    res_gb = torch.cuda.memory_reserved(self.model_slice.device) / 1024**3
                    vram_str = f"{alloc_gb:.2f} GB (reserved: {res_gb:.2f} GB)"

                logger.info(
                    "Pipeline node live & healthy — listening on %s:%d | layers [%d, %d) | active sessions: %d | VRAM=%s",
                    self.listen_host,
                    self.listen_port,
                    self.model_slice.layer_start,
                    self.model_slice.layer_end,
                    self.kv_store.active_sessions,
                    vram_str,
                )
                sys.stdout.flush()
        except asyncio.CancelledError:
            await self.stop()

    async def stop(self) -> None:
        """Shut down the node."""
        await self.kv_store.stop_eviction_loop()
        self.kv_store.clear_all()
        if self._server:
            await self._server.stop()
        if self._http_server:
            await self._http_server.stop()
        if self._next_client:
            await self._next_client.close()
        logger.info("Node stopped")


def main():
    """CLI entry point for starting a pipeline node."""
    parser = argparse.ArgumentParser(description="ShardFlow Pipeline Node")
    parser.add_argument("--model", required=True, help="Model path or HF model ID")
    parser.add_argument("--layer-start", type=int, default=None, help="First layer index (inclusive); auto-assigned if omitted")
    parser.add_argument("--layer-end", type=int, default=None, help="Last layer index (exclusive); auto-assigned if omitted")
    parser.add_argument("--host", default="0.0.0.0", help="Listen host")
    parser.add_argument("--port", type=int, default=9000, help="Listen port")
    parser.add_argument("--next-host", default=None, help="Next node host (omit for last node)")
    parser.add_argument("--next-port", type=int, default=None, help="Next node port")
    parser.add_argument("--next-node-url", default=None, help="Next node HTTP/HTTPS URL (e.g. Cloudflare tunnel URL)")
    parser.add_argument("--http-port", type=int, default=None, help="HTTP server port for incoming activations (default: None)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--registry-url", default=None, help="Topology Registry URL for auto-registration")
    parser.add_argument("--public-host", default=None, help="Public address accessible by other nodes")
    parser.add_argument("--public-port", type=int, default=None, help="Public port accessible by other nodes")
    parser.add_argument("--node-id", default=None, help="Unique node identifier")
    parser.add_argument("--draft-model", default=None, help="Draft model path or ID for speculative decoding on Node 0")
    parser.add_argument("--spec-k", type=int, default=4, help="Number of speculative draft tokens per verification step (default: 4)")
    parser.add_argument("--reg-layer-start", type=int, default=None, help="Explicit layer_start to register with registry")
    parser.add_argument("--reg-layer-end", type=int, default=None, help="Explicit layer_end to register with registry")
    parser.add_argument("--expected-nodes", type=int, default=None, help="Explicit expected cluster node count")
    parser.add_argument("--hf-model-id", default=None, help="Explicit HF repo ID for registry reporting")
    parser.add_argument("--enable-cuda-graphs", action="store_true", default=True, help="Enable CUDA Graphs (default: True)")
    parser.add_argument("--no-cuda-graphs", action="store_true", help="Disable CUDA Graphs and run in pure eager mode")
    parser.add_argument("--4bit", "--load-in-4bit", dest="load_in_4bit", action="store_true", help="Load weights in 4-bit NF4 via bitsandbytes")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    import uuid
    node_id = args.node_id or f"node-{uuid.uuid4().hex[:6]}"
    pub_host = args.public_host or args.host
    pub_port = args.public_port or args.port
    layer_start = args.layer_start
    layer_end = args.layer_end
    is_last = args.next_host is None
    next_host = args.next_host
    next_port = args.next_port

    # Auto-registration with Registry if registry_url is provided
    if args.registry_url:
        import requests
        from shardflow.registry.client import poll_for_assignment
        vram = 0.0
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        try:
            reported_model_id = args.hf_model_id or args.model
            if "/" not in reported_model_id and "qwen" in reported_model_id.lower():
                reported_model_id = "Qwen/Qwen2.5-7B-Instruct"
            reg_payload = {
                "node_id": node_id,
                "addr": pub_host,
                "port": pub_port,
                "vram_available_mb": vram,
                "vram_total_mb": vram,
                "model_id": reported_model_id,
            }
            if args.reg_layer_start is not None:
                reg_payload["layer_start"] = args.reg_layer_start
            elif layer_start is not None:
                reg_payload["layer_start"] = layer_start

            if args.reg_layer_end is not None:
                reg_payload["layer_end"] = args.reg_layer_end
            elif layer_end is not None:
                reg_payload["layer_end"] = layer_end

            if args.expected_nodes is not None:
                reg_payload["expected_nodes"] = args.expected_nodes

            resp = requests.post(
                f"{args.registry_url.rstrip('/')}/register",
                json=reg_payload,
                timeout=5.0,
            )
            if resp.status_code in (200, 201):
                data = resp.json()
                # Use registry-assigned values if we didn't specify manually
                if layer_start is None:
                    layer_start = data.get("layer_start")
                if layer_end is None or layer_end == 0:
                    layer_end = data.get("layer_end")
                is_last = data.get("is_last_node") if data.get("is_last_node") is not None else is_last
                next_host = data.get("next_node_host") or next_host
                next_port = data.get("next_node_port") or next_port
                logger.info(
                    "Registered node %s with registry -> assigned layers [%s, %s)",
                    node_id, layer_start, layer_end
                )

            # Poll until the full cluster partition is ready before loading weights
            if layer_start is None or layer_end is None or layer_end == 0:
                logger.info(
                    "Waiting for final cluster assignment — polling /assignment/%s...",
                    node_id,
                )
                assignment = poll_for_assignment(args.registry_url, node_id, timeout=120.0)
                layer_start = assignment["layer_start"]
                layer_end = assignment["layer_end"]
                is_last = assignment["is_last_node"]
                next_host = assignment.get("next_node_host") or next_host
                next_port = assignment.get("next_node_port") or next_port
                logger.info(
                    "Auto-assigned layers [%d, %d) (is_last=%s)",
                    layer_start, layer_end, is_last,
                )

        except Exception as e:
            logger.warning("Failed auto-registration with registry: %s", e)

    if layer_start is None or layer_end is None:
        logger.error("Must specify --layer-start/--layer-end or provide a valid --registry-url.")
        sys.exit(1)

    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(args.model)
    total_layers = config.num_hidden_layers
    is_last = (layer_end >= total_layers)

    # Auto-detect 4-bit NF4 from model path or flag
    use_4bit = bool(args.load_in_4bit or ("nf4" in str(args.model).lower() or "4bit" in str(args.model).lower()))

    # Load model slice
    model_slice = load_layer_slice(
        model_path=args.model,
        layer_start=layer_start,
        layer_end=layer_end,
        include_norm=is_last,
        include_lm_head=is_last,
        load_in_4bit=use_4bit,
        device=args.device,
    )

    # Create and run node
    node = PipelineNode(
        model_slice=model_slice,
        is_first_node=(layer_start == 0),
        is_last_node=is_last,
        next_node_host=next_host,
        next_node_port=next_port,
        next_node_url=args.next_node_url,
        http_port=args.http_port,
        listen_host=args.host,
        listen_port=args.port,
        enable_cuda_graphs=args.enable_cuda_graphs and not args.no_cuda_graphs,
        draft_model=args.draft_model,
        spec_k=args.spec_k,
    )

    asyncio.run(node.serve_forever())


if __name__ == "__main__":
    main()

