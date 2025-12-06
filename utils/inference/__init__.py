# utils/inference/__init__.py

"""
Inference backend abstraction layer.

Provides a unified interface for different LLM inference backends:
- HTTP: OpenAI-compatible APIs (works with vLLM server, llama.cpp server, OpenAI, etc.)
- vLLM local: In-process vLLM inference
- llama_cpp_python: In-process llama-cpp-python inference
- llama_cpp_server: Managed llama.cpp server with process lifecycle management
- Transformers local: In-process HuggingFace transformers inference

Usage:
    from utils.inference import get_backend

    # HTTP backend (OpenAI-compatible)
    backend = get_backend("http", model_name="gpt-4", base_url="https://api.openai.com/v1", api_key="...")

    # Local vLLM
    backend = get_backend("vllm", model_name="meta-llama/Llama-3-8B", tensor_parallel_size=2)

    # llama.cpp with managed server
    backend = get_backend("llama_cpp_server", model_name="/path/to/model.gguf", n_gpu_layers=-1)

    # Generate
    result = backend.generate("Hello, world!", temperature=0.7, max_tokens=100)
    results = backend.generate_many(["prompt1", "prompt2"], temperature=0.7, max_tokens=100)
"""

from .base import InferenceBackend, InferenceConfig
from .http import HTTPBackend

# Lazy imports for optional backends
_BACKEND_REGISTRY = {
    "http": ("utils.inference.http", "HTTPBackend"),
    "openai": ("utils.inference.http", "HTTPBackend"),  # alias
    "vllm": ("utils.inference.vllm_local", "VLLMLocalBackend"),
    "vllm_local": ("utils.inference.vllm_local", "VLLMLocalBackend"),  # explicit
    "vllm_http": ("utils.inference.http", "HTTPBackend"),  # explicit http mode
    # llama.cpp backends
    "llama_cpp_python": ("utils.inference.llama_cpp_python", "LlamaCppLocalBackend"),
    "llama_cpp_server": ("utils.inference.llama_cpp_server", "LlamaCppServerBackend"),
    # Legacy aliases for llama.cpp
    "llamacpp": ("utils.inference.llama_cpp_python", "LlamaCppLocalBackend"),
    "llamacpp_local": ("utils.inference.llama_cpp_python", "LlamaCppLocalBackend"),
    "llamacpp_http": ("utils.inference.http", "HTTPBackend"),
    # Transformers
    "transformers": ("utils.inference.transformers_local", "TransformersLocalBackend"),
    "hf": ("utils.inference.transformers_local", "TransformersLocalBackend"),  # alias
}


def get_backend(provider: str, **kwargs) -> InferenceBackend:
    """
    Factory function to create an inference backend.

    Args:
        provider: Backend type. One of:
            - "http" / "openai": OpenAI-compatible HTTP API
            - "vllm" / "vllm_local": In-process vLLM
            - "vllm_http": vLLM via HTTP (uses http backend)
            - "llama_cpp_python" / "llamacpp": In-process llama-cpp-python
            - "llama_cpp_server": Managed llama.cpp server process
            - "llamacpp_http": llama.cpp server via HTTP (external server)
            - "transformers" / "hf": In-process HuggingFace transformers
        **kwargs: Backend-specific configuration (model_name, api_key, etc.)

    Returns:
        An initialized InferenceBackend instance

    Raises:
        ValueError: If provider is not recognized
        ImportError: If required dependencies for the backend are not installed
    """
    provider_lower = provider.lower().strip()

    if provider_lower not in _BACKEND_REGISTRY:
        available = ", ".join(sorted(set(k for k in _BACKEND_REGISTRY.keys())))
        raise ValueError(f"Unknown provider '{provider}'. Available: {available}")

    module_path, class_name = _BACKEND_REGISTRY[provider_lower]

    # Import the module and get the class
    import importlib
    try:
        module = importlib.import_module(module_path)
        backend_class = getattr(module, class_name)
    except ImportError as e:
        raise ImportError(
            f"Failed to import backend '{provider}' from {module_path}. "
            f"Make sure required dependencies are installed. Error: {e}"
        ) from e

    return backend_class(**kwargs)


def list_backends() -> list[str]:
    """Return list of available backend names (without aliases)."""
    return ["http", "vllm", "llama_cpp_python", "llama_cpp_server", "transformers"]


__all__ = [
    "InferenceBackend",
    "InferenceConfig",
    "HTTPBackend",
    "get_backend",
    "list_backends",
]
