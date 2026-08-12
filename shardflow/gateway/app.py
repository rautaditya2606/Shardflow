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
    """Format messages list into ChatML compliant prompt."""
    prompt_parts = []
    for msg in messages:
        role = msg.role.lower()
        content = msg.content.strip()
        prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    prompt_parts.append("<|im_start|>assistant\n")
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

            try:
                async for token_text in _orchestrator.generate_stream(
                    prompt=prompt,
                    max_tokens=req.max_tokens or 100,
                    temperature=req.temperature if req.temperature is not None else 0.0,
                    top_k=req.top_k or 0,
                    top_p=req.top_p if req.top_p is not None else 1.0,
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
                logger.exception("Error during streaming generation")

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
    try:
        completion_text = await _orchestrator.generate(
            prompt=prompt,
            max_tokens=req.max_tokens or 100,
            temperature=req.temperature if req.temperature is not None else 0.0,
            top_k=req.top_k or 0,
            top_p=req.top_p if req.top_p is not None else 1.0,
            stream=False,
        )
    except Exception as e:
        logger.exception("Error during non-streaming generation: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {e}",
        )

    prompt_token_ids = _orchestrator.tokenizer.encode(prompt)
    prompt_tokens = len(prompt_token_ids)
    comp_tokens = len(_orchestrator.tokenizer.encode(completion_text)) if completion_text else 0

    return ChatCompletionResponse(
        id=completion_id,
        created=created_ts,
        model=req.model,
        choices=[
            Choice(
                index=0,
                message=ChoiceMessage(role="assistant", content=completion_text),
                finish_reason="stop",
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
