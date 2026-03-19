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
from flashserve.model.llama import LlamaForCausalLM
from flashserve.engine.inference_engine import InferenceEngine
from flashserve.serving.request import (
    CompletionRequest, CompletionResponse, CompletionChoice, CompletionUsage,
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChoice, ChatMessage
)


def create_app(
    model_name: str = "flashserve-7b",
    config: Optional[LlamaConfig] = None,
    device: str = "auto",
) -> FastAPI:
    """
    Create FastAPI application with inference engine.

    Args:
        model_name: Model name for identification
        config: Model configuration (uses tiny by default)
        device: Device to use ("cuda", "cpu", or "auto")

    Returns:
        FastAPI application
    """
    app = FastAPI(title="FlashServe", version="0.1.0")

    # Initialize model
    if config is None:
        config = LlamaConfig.tiny()

    model = LlamaForCausalLM(config)
    engine = InferenceEngine(config, model=model, device=device)

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
            choices = [
                CompletionChoice(
                    text=output,
                    index=i,
                    finish_reason="length",
                )
                for i, output in enumerate(outputs)
            ]

            return CompletionResponse(
                id=str(uuid.uuid4()),
                created=int(time.time()),
                model=request.model,
                choices=choices,
                usage=CompletionUsage(
                    prompt_tokens=0,  # Would need tokenizer for accurate count
                    completion_tokens=request.max_tokens,
                    total_tokens=request.max_tokens,
                ),
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Chat completion endpoint (OpenAI format)."""
        try:
            # Convert messages to prompt
            prompt = ""
            for msg in request.messages:
                if msg.role == "system":
                    prompt += f"System: {msg.content}\n"
                elif msg.role == "user":
                    prompt += f"User: {msg.content}\n"
                elif msg.role == "assistant":
                    prompt += f"Assistant: {msg.content}\n"

            # Generate
            output = engine.generate(
                prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=request.temperature > 0,
            )

            return ChatCompletionResponse(
                id=str(uuid.uuid4()),
                created=int(time.time()),
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=output),
                        finish_reason="length",
                    )
                ],
                usage=CompletionUsage(
                    prompt_tokens=0,
                    completion_tokens=request.max_tokens,
                    total_tokens=request.max_tokens,
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
    """Stream completions as Server-Sent Events."""
    for prompt in prompts:
        for token in engine.stream_generate(
            prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        ):
            event = {
                "choices": [
                    {
                        "text": token,
                        "index": 0,
                        "finish_reason": None,
                    }
                ],
                "created": int(time.time()),
                "model": request.model,
            }
            yield f"data: {json.dumps(event)}\n\n"

        # Final event
        yield f"data: {json.dumps({'choices': [{'finish_reason': 'stop'}]})}\n\n"
