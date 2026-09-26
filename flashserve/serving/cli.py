"""
Command line entry point.

    flashserve generate --model HuggingFaceTB/SmolLM2-135M-Instruct --prompt "Write a haiku about GPUs"
    flashserve serve --model HuggingFaceTB/SmolLM2-135M-Instruct --port 8000
"""

import argparse
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="flashserve")
    sub = ap.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate", help="Generate text from a prompt")
    gen.add_argument("--model", required=True, help="Hugging Face model id or local directory")
    gen.add_argument("--prompt", required=True)
    gen.add_argument("--max-tokens", type=int, default=128)
    gen.add_argument("--temperature", type=float, default=0.0)
    gen.add_argument("--chat", action="store_true", help="Wrap the prompt in the model's chat template")

    srv = sub.add_parser("serve", help="Run the OpenAI-compatible HTTP server")
    srv.add_argument("--model", required=True)
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8000)

    args = ap.parse_args(argv)
    from flashserve.engine.inference_engine import InferenceEngine

    engine = InferenceEngine.from_pretrained(args.model, device="auto")
    if args.cmd == "generate":
        prompt = engine.format_chat([{"role": "user", "content": args.prompt}]) if args.chat else args.prompt
        for piece in engine.stream_generate(prompt, max_tokens=args.max_tokens, temperature=args.temperature):
            sys.stdout.write(piece)
            sys.stdout.flush()
        sys.stdout.write("\n")
        return 0

    import uvicorn

    from flashserve.serving.server import create_app

    uvicorn.run(create_app(model_name=args.model, engine=engine), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
