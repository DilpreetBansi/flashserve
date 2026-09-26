"""
FastAPI serving layer with OpenAI-compatible endpoints.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import time
import uuid
import json
from typing import Optional

from flashserve.model.config import LlamaConfig
from flashserve.engine.inference_engine import InferenceEngine
from flashserve.serving.request import (
    CompletionRequest, CompletionResponse, CompletionChoice, CompletionUsage,
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChoice, ChatMessage
)


def create_app(
    model_name: str = "flashserve-tiny",
    config: Optional[LlamaConfig] = None,
    device: str = "auto",
    engine: Optional[InferenceEngine] = None,
) -> FastAPI:
    """
    Create FastAPI application with inference engine.

    Args:
        model_name: Model name for identification
        config: Model configuration (uses tiny by default); ignored if engine is given
        device: Device to use ("cuda", "cpu", or "auto")
        engine: Pre-built engine (e.g. with loaded weights and a matching tokenizer)

    Returns:
        FastAPI application
    """
    app = FastAPI(title="FlashServe", version="0.1.0")

    if engine is None:
        engine = InferenceEngine(config or LlamaConfig.tiny(), device=device)
    config = engine.config

    @app.get("/health")
    def health():
        """Health check endpoint."""
        return {
            "status": "ok",
            "model": model_name,
            "device": str(engine.device),
            "config": {
                "hidden_size": config.hidden_size,
                "num_layers": config.num_hidden_layers,
                "vocab_size": config.vocab_size,
            }
        }

    @app.post("/v1/completions")
    def completions(request: CompletionRequest) -> CompletionResponse:
        """Text completion endpoint (OpenAI format)."""
        try:
            # Handle prompt as string or list
            if isinstance(request.prompt, list):
                prompts = request.prompt
            else:
                prompts = [request.prompt]

            # Generate
            if request.stream:
                return StreamingResponse(
                    _stream_completions(engine, request, prompts),
                    media_type="text/event-stream",
                )

            # Non-streaming
            outputs = engine.generate_batch(
                prompts,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                do_sample=request.temperature > 0,
            )

            # Format response
            prompt_tokens = sum(engine.count_tokens(p) for p in prompts)
            completion_tokens = sum(engine.count_tokens(o) for o in outputs)
            choices = [
                CompletionChoice(
                    text=output,
                    index=i,
                    finish_reason="length" if engine.count_tokens(output) >= request.max_tokens else "stop",
                )
                for i, output in enumerate(outputs)
            ]

            return CompletionResponse(
                id=f"cmpl-{uuid.uuid4().hex[:24]}",
                created=int(time.time()),
                model=request.model,
                choices=choices,
                usage=CompletionUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Chat completion endpoint (OpenAI format)."""
        try:
            prompt = engine.format_chat([{"role": m.role, "content": m.content} for m in request.messages])
            output = engine.generate(
                prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=request.temperature > 0,
            )
            prompt_tokens, completion_tokens = engine.count_tokens(prompt), engine.count_tokens(output)

            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
                created=int(time.time()),
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=output),
                        finish_reason="length" if completion_tokens >= request.max_tokens else "stop",
                    )
                ],
                usage=CompletionUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/v1/models")
    def list_models():
        """List available models."""
        return {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "owned_by": "flashserve",
                    "permission": [],
                }
            ],
        }

    return app


async def _stream_completions(engine, request, prompts):
    """Stream completions as Server-Sent Events in the OpenAI format."""
    completion_id = f"cmpl-{uuid.uuid4().hex[:24]}"
    for index, prompt in enumerate(prompts):
        for token in engine.stream_generate(
            prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
        ):
            event = {
                "id": completion_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{"text": token, "index": index, "finish_reason": None}],
            }
            yield f"data: {json.dumps(event)}\n\n"

        final = {
            "id": completion_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{"text": "", "index": index, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final)}\n\n"

    yield "data: [DONE]\n\n"
