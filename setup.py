from setuptools import setup, find_packages
from app import __version__


setup(
    name="mlx-openai-server",
    url="https://github.com/cubist38/mlx-openai-server",
    author="Gia-Huy Vuong",
    author_email="cubist38@gmail.com",
    version=__version__,
    description="A high-performance API server that provides OpenAI-compatible endpoints for MLX models. Built with Python and FastAPI, it enables efficient, scalable, and user-friendly local deployment of MLX-based multimodal models with an OpenAI-compatible interface. Supports text, vision, and audio processing capabilities. Perfect for developers looking to run MLX models locally while maintaining compatibility with existing OpenAI-based applications.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(), 
    install_requires=[
        "mlx-vlm==0.3.5",
        "mlx-lm==0.28.3",
        "torchvision==0.23.0",
        "mlx-whisper==0.4.3",
        "mlx-embeddings==0.0.4",
        "fastapi==0.115.14",
        "av==16.0.1",
        "uvicorn==0.35.0",
        "Pillow==10.4.0",
        "click==8.2.1",
        "loguru==0.7.3",
        "outlines==1.1.1",
        "librosa==0.11.0",
        "openai-harmony==0.0.4",
        "json_repair==0.52.1",
        "python-multipart==0.0.20"
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "isort",
            "flake8",
        ]
    },
    entry_points={
        "console_scripts": [
            "mlx-openai-server=app.cli:cli",
        ],
    },
    python_requires=">=3.11",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
) 