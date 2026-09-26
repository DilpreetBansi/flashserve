from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="flashserve",
    version="0.2.0",
    author="Dilpreet Singh Bansi",
    description="A from-scratch LLM inference engine for Llama-architecture models (KV cache, "
    "batching, speculative decoding, OpenAI-compatible API)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DilpreetBansi/flashserve",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={"dev": ["pytest>=7.4", "httpx>=0.25", "transformers>=4.40"]},
    entry_points={"console_scripts": ["flashserve=flashserve.serving.cli:main"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
