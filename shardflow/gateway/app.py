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
from shardflow.registry.app import app as registry_app
app.include_router(registry_app.router)

# Global orchestrator reference
_orchestrator: Optional[Orchestrator] = None


def set_orchestrator(orchestrator: Orchestrator) -> None:
    global _orchestrator
    _orchestrator = orchestrator


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
        port = os.getenv("PORT", "8000")
        default_url = f"http://127.0.0.1:{port}"
        registry_url = os.getenv("SHARDFLOW_REGISTRY_URL", default_url)
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
            # Initial chunk with role
            role_chunk = ChatCompletionChunk(
                id=completion_id,
                created=created_ts,
                model=req.model,
                choices=[ChunkChoice(index=0, delta=DeltaMessage(role="assistant"), finish_reason=None)],
            )
            yield f"data: {json.dumps(role_chunk.model_dump())}\n\n"

            # Stream tokens
            # We can capture tokens as orchestrator generates
            # For streaming, we generate tokens incrementally
            try:
                # Custom async token yield generator calling orchestrator logic
                session_id = str(uuid.uuid4())
                input_ids = _orchestrator.tokenizer(prompt, return_tensors="pt")["input_ids"]
                prompt_len = input_ids.shape[1]

                hidden_states = _orchestrator._embed(input_ids)
                from shardflow.transport.protocol import TensorMessage, MessageType
                from shardflow.orchestrator.sampler import sample_next_token

                msg = TensorMessage(
                    msg_type=MessageType.ACTIVATION,
                    session_id=session_id,
                    tensor=hidden_states.cpu(),
                    temperature=req.temperature,
                    top_k=req.top_k,
                    top_p=req.top_p,
                    sample_on_node=True,
                )
                response = await _orchestrator._node0_client.send_recv(msg)
                if response.msg_type == MessageType.TOKEN_ID:
                    next_token = response.token_id
                elif response.msg_type == MessageType.LOGITS:
                    logits = response.tensor[0, -1, :]
                    next_token = sample_next_token(logits, temperature=req.temperature, top_k=req.top_k, top_p=req.top_p)
                else:
                    raise RuntimeError(f"Expected TOKEN_ID or LOGITS, got {response.msg_type}")

                token_text = _orchestrator.tokenizer.decode([next_token])
                chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=created_ts,
                    model=req.model,
                    choices=[ChunkChoice(index=0, delta=DeltaMessage(content=token_text), finish_reason=None)],
                )
                yield f"data: {json.dumps(chunk.model_dump())}\n\n"

                for _ in range(1, req.max_tokens):
                    if next_token == _orchestrator.tokenizer.eos_token_id:
                        break

                    # Check client disconnect
                    if await raw_request.is_disconnected():
                        logger.info("Client disconnected during stream for session %s", session_id)
                        break

                    import torch
                    token_ids = torch.tensor([[next_token]], dtype=torch.long)
                    hidden_states = _orchestrator._embed(token_ids)
                    msg = TensorMessage(
                        msg_type=MessageType.ACTIVATION,
                        session_id=session_id,
                        tensor=hidden_states.cpu(),
                        temperature=req.temperature,
                        top_k=req.top_k,
                        top_p=req.top_p,
                        sample_on_node=True,
                    )
                    response = await _orchestrator._node0_client.send_recv(msg)
                    if response.msg_type == MessageType.TOKEN_ID:
                        next_token = response.token_id
                    elif response.msg_type == MessageType.LOGITS:
                        logits = response.tensor[0, -1, :]
                        next_token = sample_next_token(logits, temperature=req.temperature, top_k=req.top_k, top_p=req.top_p)
                    else:
                        raise RuntimeError(f"Expected TOKEN_ID or LOGITS, got {response.msg_type}")

                    token_text = _orchestrator.tokenizer.decode([next_token])
                    chunk = ChatCompletionChunk(
                        id=completion_id,
                        created=created_ts,
                        model=req.model,
                        choices=[ChunkChoice(index=0, delta=DeltaMessage(content=token_text), finish_reason=None)],
                    )
                    yield f"data: {json.dumps(chunk.model_dump())}\n\n"

                # Final chunk
                final_chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=created_ts,
                    model=req.model,
                    choices=[ChunkChoice(index=0, delta=DeltaMessage(), finish_reason="stop")],
                )
                yield f"data: {json.dumps(final_chunk.model_dump())}\n\n"
                yield "data: [DONE]\n\n"

            except Exception as e:
                logger.exception("Error during SSE streaming")
            finally:
                # Cleanup session on finish or disconnect
                clear_msg = TensorMessage(msg_type=MessageType.CLEAR, session_id=session_id, tensor=None)
                try:
                    await _orchestrator._node0_client.send(clear_msg)
                except Exception:
                    pass

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
    """Auto-initialize orchestrator if environment variables are set."""
    global _orchestrator
    import os
    registry_url = os.getenv("SHARDFLOW_REGISTRY_URL")
    model_path = os.getenv("SHARDFLOW_MODEL_PATH", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    if _orchestrator is None and registry_url:
        logger.info("Auto-initializing Orchestrator from environment (Registry: %s, Model: %s)", registry_url, model_path)
        try:
            orch = Orchestrator(
                model_path=model_path,
                registry_url=registry_url,
                device="cpu",
            )
            await orch.initialize()
            set_orchestrator(orch)
            logger.info("Orchestrator successfully initialized on startup.")
        except Exception as e:
            logger.warning("Could not auto-initialize Orchestrator on startup (waiting for active nodes): %s", e)


@app.get("/")
def read_root():
    return {
        "service": "ShardFlow OpenAI-Compatible API Gateway",
        "status": "online",
        "version": "0.1.0",
        "docs_url": "/docs",
        "health_url": "/health",
        "metrics_url": "/metrics",
    }


@app.get("/health")
def health():
    return {"status": "ok", "orchestrator_ready": _orchestrator is not None}


@app.get("/metrics")
def get_metrics():
    """Return runtime metrics summary."""
    from shardflow.orchestrator.metrics import metrics
    return metrics.get_summary()
