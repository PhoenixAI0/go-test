from setuptools import setup, find_packages

setup(
    name="go_rl",
    version="0.1.0",
    description="Go RL training utilities with KataGo-based rewards",
    author="Go RL Team",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.4.0",
        "transformers>=4.45.0",
        "datasets>=3.0.0",
        "trl>=0.11.0",
        "accelerate>=0.34.0",
        "sgfmill>=1.1.1",
        "pyyaml>=6.0.1",
        "requests>=2.31.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "ruff>=0.6.0",
        ],
        "wandb": [
            "wandb>=0.18.0",
        ],
    },
)
