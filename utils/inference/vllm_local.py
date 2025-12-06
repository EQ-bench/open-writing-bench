# utils/inference/vllm_local.py

"""
Local vLLM backend for in-process inference.

Requires: pip install vllm
"""

import logging
import os
from typing import Any, Optional

from .base import InferenceBackend

logger = logging.getLogger(__name__)


class VLLMLocalBackend(InferenceBackend):
    """
    In-process vLLM inference backend.

    Parallelism: Uses vLLM's native batching via generate() with multiple prompts.
    vLLM handles GPU parallelism internally (tensor parallel, continuous batching).
    """

    # Allowed environment variables that can be set via ENV_VARS config
    ALLOWED_ENV_VARS = {
        "VLLM_ATTENTION_BACKEND",
        "VLLM_USE_TRITON_FLASH_ATTN",
        "VLLM_USE_V1",
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
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize vLLM backend.

        Args:
            model_name: HuggingFace model name or path
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use
            max_model_len: Maximum sequence length (None = auto)
            dtype: Model dtype ("auto", "float16", "bfloat16", "float32")
            quantization: Quantization method (None, "awq", "gptq", "squeezellm")
            seed: Random seed for reproducibility
            **kwargs: Additional vLLM engine args. Special keys:
                ENV_VARS: dict of environment variables to set before loading vLLM.
                    Only allowed vars: VLLM_ATTENTION_BACKEND, VLLM_USE_TRITON_FLASH_ATTN,
                    VLLM_USE_V1.

        Note:
            trust_remote_code is always set to False for security.
        """
        # Set environment variables BEFORE importing vLLM
        env_vars = kwargs.pop("ENV_VARS", None)
        if env_vars:
            self._set_env_vars(env_vars)

        # Filter out trust_remote_code if passed (always disabled for security)
        kwargs.pop("trust_remote_code", None)
        super().__init__(model_name, **kwargs)

        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise ImportError(
                "vLLM is not installed. Install with: pip install vllm"
            ) from e

        self._SamplingParams = SamplingParams

        # Collect engine args
        # Note: trust_remote_code is always False for security
        engine_kwargs = {
            "model": model_name,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "dtype": dtype,
            "trust_remote_code": False,
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

        logger.info(f"Loading vLLM model: {model_name}")
        logger.debug(f"vLLM engine args: {engine_kwargs}")

        self._llm = LLM(**engine_kwargs)

        logger.info(f"vLLM model loaded: {model_name}")

    def _set_env_vars(self, env_vars: dict[str, str]) -> None:
        """Set allowed environment variables before vLLM import."""
        for key, value in env_vars.items():
            if key not in self.ALLOWED_ENV_VARS:
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

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a single prompt."""
        sampling_params = self._build_sampling_params(**kwargs)
        outputs = self._llm.generate([prompt], sampling_params)

        if not outputs or not outputs[0].outputs:
            raise RuntimeError("vLLM returned empty output")

        return outputs[0].outputs[0].text.strip()

    def generate_many(self, prompts: list[str], **kwargs) -> list[str]:
        """
        Generate text from multiple prompts using vLLM's native batching.

        vLLM handles continuous batching internally for optimal throughput.
        """
        if not prompts:
            return []

        sampling_params = self._build_sampling_params(**kwargs)
        outputs = self._llm.generate(prompts, sampling_params)

        results = []
        for output in outputs:
            if output.outputs:
                results.append(output.outputs[0].text.strip())
            else:
                results.append("[ERROR] Empty output")

        return results

    def close(self) -> None:
        """Release vLLM resources."""
        # vLLM doesn't have an explicit cleanup method
        # but we can delete the reference to allow GC
        if hasattr(self, '_llm'):
            del self._llm
        logger.debug("VLLMLocalBackend closed")
