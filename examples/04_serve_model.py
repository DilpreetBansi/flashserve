#!/usr/bin/env python3
"""
Launch FastAPI server with OpenAI-compatible API.

Run with:
    python examples/04_serve_model.py

Then test with:
    curl -X POST http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "flashserve-tiny",
            "prompt": "The future of AI is",
            "max_tokens": 50
        }'
"""

import uvicorn
from flashserve.model.config import LlamaConfig
from flashserve.serving.server import create_app


def main():
    # Create FastAPI app with tiny model
    config = LlamaConfig.tiny()
    app = create_app(
        model_name="flashserve-tiny",
        config=config,
        device="cpu",
    )

    print("Starting FlashServe API server...")
    print("Available endpoints:")
    print("  POST /v1/completions        - Text completion")
    print("  POST /v1/chat/completions   - Chat completion")
    print("  GET  /health                - Health check")
    print("  GET  /v1/models             - List models")
    print()
    print("Server running at http://localhost:8000")
    print("API docs at http://localhost:8000/docs")
    print()

    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )


if __name__ == "__main__":
    main()
