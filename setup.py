from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="flashserve",
    version="0.1.0",
    author="FlashServe Contributors",
    author_email="bansidil@my.yorku.ca",
    description="High-Performance LLM Inference Engine with Memory-Efficient Attention",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DilpreetBansi/flashserve",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "flashserve=flashserve.serving.cli:main",
        ],
    },
    include_package_data=True,
)
