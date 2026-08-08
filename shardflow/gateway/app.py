"""
Layer 2 — API Gateway (FastAPI)

Exposes an OpenAI-compatible /v1/chat/completions endpoint.
Delegates generation requests to the Inference Orchestrator.
Supports standard JSON responses and SSE streaming (`stream=True`).
Includes session cancellation endpoint DELETE /v1/sessions/{session_id}.
"""

import argparse
import asyncio
import json
import logging
import time
import uuid
from typing import Optional, AsyncGenerator

from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from shardflow.gateway.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    UsageInfo,
    ChatCompletionChunk,
    ChunkChoice,
    DeltaMessage,
)
from shardflow.orchestrator.orchestrator import Orchestrator

logger = logging.getLogger(__name__)

app = FastAPI(
    title="ShardFlow OpenAI API Gateway",
    description="OpenAI-compatible LLM endpoint for distributed inference across N nodes",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Topology Registry endpoints (/register, /topology, /heartbeat) directly
from shardflow.registry.app import router as registry_router
app.include_router(registry_router)


# Global orchestrator and stream receiver references
_orchestrator: Optional[Orchestrator] = None
_stream_receiver: Optional[object] = None


def set_orchestrator(orchestrator: Orchestrator) -> None:
    global _orchestrator
    _orchestrator = orchestrator


async def get_stream_receiver():
    """Lazily start and return Gateway StreamReceiverServer."""
    global _stream_receiver
    if _stream_receiver is None:
        from shardflow.transport.connection import StreamReceiverServer
        import os
        port = int(os.getenv("SHARDFLOW_STREAM_PORT", "0"))
        receiver = StreamReceiverServer(host="0.0.0.0", port=port)
        await receiver.start()
        _stream_receiver = receiver
    return _stream_receiver


def _format_prompt(messages: list) -> str:
    """Format messages list into a plain text prompt."""
    prompt_parts = []
    for msg in messages:
        role = msg.role.lower()
        content = msg.content.strip()
        if role == "system":
            prompt_parts.append(f"System: {content}\n")
        elif role == "user":
            prompt_parts.append(f"User: {content}\n")
        elif role == "assistant":
            prompt_parts.append(f"Assistant: {content}\n")
    prompt_parts.append("Assistant:")
    return "\n".join(prompt_parts)


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, raw_request: Request):
    """OpenAI-compatible chat completions endpoint."""
    global _orchestrator
    if _orchestrator is None or _orchestrator._node0_client is None:
        import os
        # None = use embedded in-memory registry (avoids HTTP loopback deadlock on Render)
        registry_url = os.getenv("SHARDFLOW_REGISTRY_URL")
        model_path = req.model or os.getenv("SHARDFLOW_MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct")
        try:
            logger.info("Auto-connecting Orchestrator to active Node 0 (Model: %s, Registry: %s)...", model_path, registry_url)
            orch = Orchestrator(model_path=model_path, registry_url=registry_url, device="cpu")
            await orch.initialize()
            _orchestrator = orch
        except Exception as e:
            logger.warning("Could not auto-initialize Orchestrator: %s", e)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Orchestrator not initialized or nodes not ready: {e}",
            )

    # Format messages to prompt string
    if hasattr(_orchestrator.tokenizer, "apply_chat_template"):
        try:
            msg_dicts = [{"role": m.role, "content": m.content} for m in req.messages]
            prompt = _orchestrator.tokenizer.apply_chat_template(
                msg_dicts, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = _format_prompt(req.messages)
    else:
        prompt = _format_prompt(req.messages)

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_ts = int(time.time())

    # Handle streaming SSE response
    if req.stream:
        async def event_generator() -> AsyncGenerator[str, None]:
            role_chunk = ChatCompletionChunk(
                id=completion_id,
                created=created_ts,
                model=req.model,
                choices=[ChunkChoice(index=0, delta=DeltaMessage(role="assistant"), finish_reason=None)],
            )
            yield f"data: {json.dumps(role_chunk.model_dump())}\n\n"

            # v2 Data Plane stream path: send START_SESSION to Node 0 and stream directly from StreamReceiver
            session_id = str(uuid.uuid4())
            stream_receiver = await get_stream_receiver()
            stream_q = stream_receiver.register_session(session_id)

            try:
                import os
                stream_host = os.getenv("SHARDFLOW_GATEWAY_HOST", "127.0.0.1")
                prompt_token_ids = _orchestrator.tokenizer.encode(prompt)
                eos_id = getattr(_orchestrator.tokenizer, "eos_token_id", None)

                from shardflow.transport.protocol import TensorMessage, MessageType
                start_msg = TensorMessage(
                    msg_type=MessageType.START_SESSION,
                    session_id=session_id,
                    prompt_tokens=prompt_token_ids,
                    max_tokens=req.max_tokens or 100,
                    temperature=req.temperature or 0.0,
                    top_k=req.top_k or 0,
                    top_p=req.top_p or 1.0,
                    eos_token_id=eos_id,
                    stream_back_host=stream_host,
                    stream_back_port=stream_receiver.bound_port,
                )
                await _orchestrator._node0_client.send(start_msg)

                while True:
                    if await raw_request.is_disconnected():
                        clear_msg = TensorMessage(msg_type=MessageType.CLEAR, session_id=session_id)
                        try:
                            await _orchestrator._node0_client.send(clear_msg)
                        except Exception:
                            pass
                        break

                    try:
                        token_msg = await asyncio.wait_for(stream_q.get(), timeout=30.0)
                    except asyncio.TimeoutError:
                        break

                    if token_msg.is_eos or (token_msg.finish_reason is not None and token_msg.finish_reason != ""):
                        break

                    if token_msg.token_id is not None:
                        token_text = _orchestrator.tokenizer.decode([token_msg.token_id])
                        chunk = ChatCompletionChunk(
                            id=completion_id,
                            created=created_ts,
                            model=req.model,
                            choices=[ChunkChoice(index=0, delta=DeltaMessage(content=token_text), finish_reason=None)],
                        )
                        yield f"data: {json.dumps(chunk.model_dump())}\n\n"

            except Exception:
                logger.exception("Error during v2 SSE streaming — falling back to generator")
                try:
                    async for token_text in _orchestrator.generate_stream(
                        prompt=prompt,
                        max_tokens=req.max_tokens,
                        temperature=req.temperature,
                        top_k=req.top_k,
                        top_p=req.top_p,
                    ):
                        if await raw_request.is_disconnected():
                            break
                        chunk = ChatCompletionChunk(
                            id=completion_id,
                            created=created_ts,
                            model=req.model,
                            choices=[ChunkChoice(index=0, delta=DeltaMessage(content=token_text), finish_reason=None)],
                        )
                        yield f"data: {json.dumps(chunk.model_dump())}\n\n"
                except Exception:
                    logger.exception("Fallback stream also encountered error")
            finally:
                stream_receiver.unregister_session(session_id)

            final_chunk = ChatCompletionChunk(
                id=completion_id,
                created=created_ts,
                model=req.model,
                choices=[ChunkChoice(index=0, delta=DeltaMessage(), finish_reason="stop")],
            )
            yield f"data: {json.dumps(final_chunk.model_dump())}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # Non-streaming JSON response
    from shardflow.orchestrator.orchestrator import PartialGenerationError

    finish_reason = "stop"
    try:
        completion_text = await _orchestrator.generate(
            prompt=prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            stream=False,
        )
    except PartialGenerationError as e:
        logger.warning("Returning partial generation text due to node error: %s", e)
        completion_text = e.partial_text
        finish_reason = "node_failure"

    prompt_tokens = len(_orchestrator.tokenizer.encode(prompt))
    comp_tokens = len(_orchestrator.tokenizer.encode(completion_text)) if completion_text else 0

    return ChatCompletionResponse(
        id=completion_id,
        created=created_ts,
        model=req.model,
        choices=[
            Choice(
                index=0,
                message=ChoiceMessage(role="assistant", content=completion_text),
                finish_reason=finish_reason,
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=comp_tokens,
            total_tokens=prompt_tokens + comp_tokens,
        ),
    )


@app.delete("/v1/sessions/{session_id}")
async def cancel_session(session_id: str):
    """Explicit session cancellation endpoint."""
    if _orchestrator and _orchestrator._node0_client:
        from shardflow.transport.protocol import TensorMessage, MessageType
        clear_msg = TensorMessage(msg_type=MessageType.CLEAR, session_id=session_id, tensor=None)
        await _orchestrator._node0_client.send(clear_msg)
        return {"status": "cancelled", "session_id": session_id}
    raise HTTPException(status_code=503, detail="Orchestrator not ready")


@app.on_event("startup")
async def startup_event():
    """
    Intentionally lightweight — do NOT initialize the Orchestrator here.

    Reason: Render free tier has a 512 MB RAM limit. Calling
    AutoTokenizer.from_pretrained() during startup downloads and decompresses
    the tokenizer vocab to /tmp (~500-600 MB peak), which kills the process.

    The Orchestrator is initialized lazily on the first POST /v1/chat/completions
    request, by which time Colab nodes will have registered themselves via /register.
    This keeps idle RAM at ~30-50 MB.
    """
    import os
    registry_url = os.getenv("SHARDFLOW_REGISTRY_URL", "(not set)")
    model_path = os.getenv("SHARDFLOW_MODEL_PATH", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    logger.info(
        "ShardFlow Gateway started. Orchestrator will initialize lazily on first request. "
        "Registry: %s | Model: %s",
        registry_url, model_path,
    )


@app.api_route("/", methods=["GET", "HEAD"])
def read_root():
    return {
        "service": "ShardFlow OpenAI-Compatible API Gateway",
        "status": "online",
        "version": "0.1.0",
        "docs_url": "/docs",
        "health_url": "/health",
        "metrics_url": "/metrics",
    }


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {"status": "ok", "orchestrator_ready": _orchestrator is not None}


@app.get("/metrics")
def get_metrics():
    """Return runtime metrics summary."""
    from shardflow.orchestrator.metrics import metrics
    return metrics.get_summary()
