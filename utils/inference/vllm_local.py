# utils/inference/vllm_local.py

"""
Local vLLM backend for in-process inference using AsyncLLMEngine.

Requires: pip install vllm

This backend uses vLLM's AsyncLLMEngine for true concurrent request handling.
Multiple generate() calls from different threads are processed concurrently
via the async engine's continuous batching.
"""

import asyncio
import logging
import os
import subprocess
import threading
import uuid
from typing import Any, Optional

from .base import InferenceBackend
from .trust_remote_code import should_trust_remote_code

logger = logging.getLogger(__name__)


def _get_gpu_count() -> int:
    """Get the number of available GPUs."""
    try:
        import torch
        return torch.cuda.device_count() or 1
    except ImportError:
        pass

    # Fallback: check CUDA_VISIBLE_DEVICES
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_devices:
        return len([d for d in cuda_devices.split(",") if d.strip()])

    # Fallback: try nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return len(result.stdout.strip().split("\n"))
    except Exception:
        pass

    return 1


class VLLMLocalBackend(InferenceBackend):
    """
    In-process vLLM inference backend using AsyncLLMEngine.

    Parallelism: Uses vLLM's AsyncLLMEngine for true concurrent request processing.
    Multiple requests are batched together by the engine for optimal GPU utilization.
    """

    # When True, only params in ALLOWED_ENV_VARS and KNOWN_INIT_PARAMS are accepted
    RESTRICT_TO_ALLOWLIST = False

    # Allowed environment variables that can be set via ENV_VARS config
    ALLOWED_ENV_VARS = {
        "VLLM_ATTENTION_BACKEND",
        "VLLM_USE_TRITON_FLASH_ATTN",
        "VLLM_USE_V1",
        "VLLM_DISABLE_FLASHINFER",
        "VLLM_USE_FLASHINFER_SAMPLER",
        "VLLM_USE_TRTLLM_ATTENTION",
        "CUDA_VISIBLE_DEVICES",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
    }

    KNOWN_INIT_PARAMS = {
        "model_name",
        # Environment variables (set before vLLM import)
        "ENV_VARS",
        # vLLM engine args
        "tensor_parallel_size", "pipeline_parallel_size",
        "gpu_memory_utilization", "max_model_len",
        "dtype", "quantization", "load_format",
        "tokenizer", "tokenizer_mode",
        "revision", "download_dir", "seed",
        "enforce_eager", "max_num_seqs", "max_num_batched_tokens",
        "enable_prefix_caching", "disable_log_stats",
        # Ignored for security (always False)
        "trust_remote_code",
        # Concurrency settings
        "max_concurrent",
    }

    KNOWN_GEN_PARAMS = {
        "temperature", "max_tokens", "top_p", "top_k",
        "min_p", "repetition_penalty", "stop",
        "presence_penalty", "frequency_penalty",
        "use_beam_search", "best_of", "length_penalty",
        "skip_special_tokens", "spaces_between_special_tokens",
    }

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: Optional[int] = None,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        seed: Optional[int] = None,
        max_concurrent: int = 32,
        **kwargs
    ):
        """
        Initialize vLLM backend with AsyncLLMEngine.

        Args:
            model_name: HuggingFace model name or path
            tensor_parallel_size: Number of GPUs for tensor parallelism (None = all available)
            gpu_memory_utilization: Fraction of GPU memory to use
            max_model_len: Maximum sequence length (None = auto)
            dtype: Model dtype ("auto", "float16", "bfloat16", "float32")
            quantization: Quantization method (None, "awq", "gptq", "squeezellm")
            seed: Random seed for reproducibility
            max_concurrent: Max concurrent requests for generate_many
            **kwargs: Additional vLLM engine args. Special keys:
                ENV_VARS: dict of environment variables to set before loading vLLM.

        Note:
            trust_remote_code is always set to False for security.
        """
        # Set environment variables BEFORE importing vLLM
        env_vars = kwargs.pop("ENV_VARS", None)
        if env_vars:
            self._set_env_vars(env_vars)

        # Default tensor_parallel_size to number of available GPUs
        if tensor_parallel_size is None:
            tensor_parallel_size = _get_gpu_count()
            logger.info(f"Auto-detected {tensor_parallel_size} GPU(s) for tensor parallelism")

        # Determine trust_remote_code based on allowlist
        self._trust_remote_code = should_trust_remote_code(model_name)
        if self._trust_remote_code:
            logger.info(f"Enabling trust_remote_code for model: {model_name}")
        kwargs.pop("trust_remote_code", None)  # Ignore user-provided value
        super().__init__(model_name, **kwargs)

        self.max_concurrent = max_concurrent

        try:
            from vllm import SamplingParams
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.engine.async_llm_engine import AsyncLLMEngine
        except ImportError as e:
            raise ImportError(
                "vLLM is not installed. Install with: pip install vllm"
            ) from e

        self._SamplingParams = SamplingParams

        # Collect engine args
        engine_kwargs = {
            "model": model_name,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "dtype": dtype,
            "trust_remote_code": self._trust_remote_code,
            "disable_log_requests": True,
            "max_log_len": 0,
        }

        if max_model_len is not None:
            engine_kwargs["max_model_len"] = max_model_len
        if quantization is not None:
            engine_kwargs["quantization"] = quantization
        if seed is not None:
            engine_kwargs["seed"] = seed

        # Pass through additional engine kwargs
        for key in kwargs:
            if key in self.KNOWN_INIT_PARAMS and key not in engine_kwargs:
                engine_kwargs[key] = kwargs[key]

        logger.info(f"Loading vLLM async engine: {model_name}")
        logger.debug(f"vLLM engine args: {engine_kwargs}")

        # Create async engine
        engine_args = AsyncEngineArgs(**engine_kwargs)

        # Create event loop for async operations
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._loop_thread.start()

        # Initialize engine in the event loop
        future = asyncio.run_coroutine_threadsafe(
            AsyncLLMEngine.from_engine_args(engine_args),
            self._loop
        )
        self._engine = future.result()

        logger.info(f"vLLM async engine loaded: {model_name}")

    def _run_event_loop(self):
        """Run the event loop in a background thread."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _set_env_vars(self, env_vars: dict[str, str]) -> None:
        """Set environment variables before vLLM import."""
        for key, value in env_vars.items():
            if self.RESTRICT_TO_ALLOWLIST and key not in self.ALLOWED_ENV_VARS:
                logger.warning(
                    f"VLLMLocalBackend: Ignoring disallowed env var: {key}. "
                    f"Allowed: {self.ALLOWED_ENV_VARS}"
                )
                continue
            logger.debug(f"Setting env var: {key}={value}")
            os.environ[key] = str(value)

    def _build_sampling_params(self, **kwargs) -> Any:
        """Build vLLM SamplingParams from kwargs."""
        params = {}

        # Map common params
        param_mapping = {
            "temperature": "temperature",
            "max_tokens": "max_tokens",
            "top_p": "top_p",
            "top_k": "top_k",
            "min_p": "min_p",
            "repetition_penalty": "repetition_penalty",
            "stop": "stop",
            "presence_penalty": "presence_penalty",
            "frequency_penalty": "frequency_penalty",
        }

        for src, dst in param_mapping.items():
            if src in kwargs and kwargs[src] is not None:
                params[dst] = kwargs[src]

        # Pass through other known gen params
        for key in kwargs:
            if key in self.KNOWN_GEN_PARAMS and key not in params:
                if kwargs[key] is not None:
                    params[key] = kwargs[key]

        # Warn about unknown params
        unknown = set(kwargs.keys()) - self.KNOWN_GEN_PARAMS
        if unknown:
            logger.debug(f"VLLMLocalBackend: ignoring unknown gen params: {unknown}")

        return self._SamplingParams(**params)

    async def _generate_async(self, prompt: str, request_id: str, sampling_params: Any) -> str:
        """Generate text asynchronously using the async engine."""
        final_output = None

        async for output in self._engine.generate(prompt, sampling_params, request_id):
            final_output = output

        if final_output is None or not final_output.outputs:
            raise RuntimeError("vLLM returned empty output")

        return final_output.outputs[0].text.strip()

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a single prompt."""
        sampling_params = self._build_sampling_params(**kwargs)
        request_id = str(uuid.uuid4())

        # Submit to the event loop and wait for result
        future = asyncio.run_coroutine_threadsafe(
            self._generate_async(prompt, request_id, sampling_params),
            self._loop
        )
        return future.result()

    def generate_many(self, prompts: list[str], **kwargs) -> list[str]:
        """
        Generate text from multiple prompts using concurrent async requests.

        All requests are submitted to the AsyncLLMEngine concurrently,
        allowing vLLM to batch them together for optimal GPU utilization.
        """
        if not prompts:
            return []

        if len(prompts) == 1:
            return [self.generate(prompts[0], **kwargs)]

        sampling_params = self._build_sampling_params(**kwargs)

        async def generate_all():
            tasks = []
            for i, prompt in enumerate(prompts):
                request_id = f"{uuid.uuid4()}-{i}"
                tasks.append(self._generate_async(prompt, request_id, sampling_params))
            return await asyncio.gather(*tasks, return_exceptions=True)

        # Submit all requests concurrently
        future = asyncio.run_coroutine_threadsafe(generate_all(), self._loop)
        results = future.result()

        # Process results, converting exceptions to error strings
        processed = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error generating prompt {i}: {result}")
                processed.append(f"[ERROR] {result}")
            else:
                processed.append(result)

        return processed

    def close(self) -> None:
        """Release vLLM resources and stop the event loop."""
        if hasattr(self, '_loop') and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            if hasattr(self, '_loop_thread'):
                self._loop_thread.join(timeout=5)

        if hasattr(self, '_engine'):
            del self._engine

        logger.debug("VLLMLocalBackend closed")
