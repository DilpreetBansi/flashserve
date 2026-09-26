"""
Run a real checkpoint (SmolLM2-135M-Instruct, ~270 MB download) and stream a chat reply.

    python examples/05_real_model.py
"""

from flashserve import InferenceEngine

engine = InferenceEngine.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct", device="cpu")
prompt = engine.format_chat([{"role": "user", "content": "Explain what a KV cache does in two sentences."}])

for piece in engine.stream_generate(prompt, max_tokens=80, temperature=0.0):
    print(piece, end="", flush=True)
print()
print(engine.get_stats())
