# utils/inference/llamacpp_local.py

"""
Local llama.cpp backend for in-process inference via llama-cpp-python.

Requires: pip install llama-cpp-python
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

from .base import InferenceBackend

logger = logging.getLogger(__name__)


class LlamaCppLocalBackend(InferenceBackend):
    """
    In-process llama.cpp inference via llama-cpp-python.

    Parallelism: llama.cpp is single-threaded per model instance.
    For generate_many, we use sequential generation (safest) or
    optionally a process pool for true parallelism with multiple model copies.

    Note: For high-throughput scenarios, consider using the HTTP backend
    with llama-server which handles batching more efficiently.
    """

    KNOWN_INIT_PARAMS = {
        "model_name",  # path to GGUF file
        # llama.cpp model params
        "n_ctx", "n_batch", "n_threads", "n_threads_batch",
        "n_gpu_layers", "main_gpu", "tensor_split",
        "vocab_only", "use_mmap", "use_mlock",
        "seed", "rope_freq_base", "rope_freq_scale",
        "mul_mat_q", "logits_all", "embedding",
        "offload_kqv", "flash_attn",
        "verbose",
        # Parallelism
        "max_concurrent",
    }

    KNOWN_GEN_PARAMS = {
        "temperature", "max_tokens", "top_p", "top_k",
        "min_p", "repeat_penalty", "stop",
        "presence_penalty", "frequency_penalty",
        "mirostat_mode", "mirostat_tau", "mirostat_eta",
        "tfs_z", "typical_p",
    }

    def __init__(
        self,
        model_name: str,  # path to GGUF file
        n_ctx: int = 4096,
        n_batch: int = 512,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = -1,  # -1 = all layers on GPU
        seed: int = -1,
        verbose: bool = False,
        max_concurrent: int = 1,  # sequential by default (safest)
        **kwargs
    ):
        """
        Initialize llama.cpp backend.

        Args:
            model_name: Path to GGUF model file
            n_ctx: Context window size
            n_batch: Batch size for prompt processing
            n_threads: Number of CPU threads (None = auto)
            n_gpu_layers: Layers to offload to GPU (-1 = all)
            seed: Random seed (-1 = random)
            verbose: Enable llama.cpp verbose output
            max_concurrent: Max concurrent generations (1 = sequential)
            **kwargs: Additional llama-cpp-python params
        """
        super().__init__(model_name, **kwargs)

        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise ImportError(
                "llama-cpp-python is not installed. Install with: "
                "pip install llama-cpp-python"
            ) from e

        self.max_concurrent = max_concurrent

        # Collect model init args
        model_kwargs = {
            "model_path": model_name,
            "n_ctx": n_ctx,
            "n_batch": n_batch,
            "n_gpu_layers": n_gpu_layers,
            "seed": seed,
            "verbose": verbose,
        }

        if n_threads is not None:
            model_kwargs["n_threads"] = n_threads

        # Pass through additional known init params
        passthrough = {
            "n_threads_batch", "main_gpu", "tensor_split",
            "vocab_only", "use_mmap", "use_mlock",
            "rope_freq_base", "rope_freq_scale",
            "mul_mat_q", "logits_all", "embedding",
            "offload_kqv", "flash_attn",
        }
        for key in passthrough:
            if key in kwargs and kwargs[key] is not None:
                model_kwargs[key] = kwargs[key]

        logger.info(f"Loading llama.cpp model: {model_name}")
        logger.debug(f"llama.cpp args: {model_kwargs}")

        self._llm = Llama(**model_kwargs)

        logger.info(f"llama.cpp model loaded: {model_name}")

    def _build_generate_kwargs(self, **kwargs) -> dict[str, Any]:
        """Build kwargs for llama.cpp generate call."""
        gen_kwargs = {}

        # Map common params (note: llama-cpp uses repeat_penalty, not repetition_penalty)
        param_mapping = {
            "temperature": "temperature",
            "max_tokens": "max_tokens",
            "top_p": "top_p",
            "top_k": "top_k",
            "min_p": "min_p",
            "repeat_penalty": "repeat_penalty",
            "repetition_penalty": "repeat_penalty",  # alias
            "stop": "stop",
            "presence_penalty": "presence_penalty",
            "frequency_penalty": "frequency_penalty",
        }

        for src, dst in param_mapping.items():
            if src in kwargs and kwargs[src] is not None:
                gen_kwargs[dst] = kwargs[src]

        # Pass through other known gen params
        passthrough = {"mirostat_mode", "mirostat_tau", "mirostat_eta", "tfs_z", "typical_p"}
        for key in passthrough:
            if key in kwargs and kwargs[key] is not None:
                gen_kwargs[key] = kwargs[key]

        # Warn about unknown params
        all_known = set(param_mapping.keys()) | passthrough
        unknown = set(kwargs.keys()) - all_known
        if unknown:
            logger.debug(f"LlamaCppLocalBackend: ignoring unknown gen params: {unknown}")

        return gen_kwargs

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a single prompt."""
        gen_kwargs = self._build_generate_kwargs(**kwargs)

        # llama-cpp-python create_completion returns a dict
        output = self._llm.create_completion(
            prompt=prompt,
            **gen_kwargs
        )

        if isinstance(output, dict):
            text = output.get("choices", [{}])[0].get("text", "")
        else:
            # Streaming mode returns a generator (we don't use it)
            raise RuntimeError("Unexpected streaming output from llama.cpp")

        return text.strip()

    def generate_many(self, prompts: list[str], **kwargs) -> list[str]:
        """
        Generate text from multiple prompts.

        Uses sequential generation by default (max_concurrent=1) since
        llama.cpp is single-threaded. Can use thread pool for I/O overlap
        but true parallelism requires multiple model instances.
        """
        if not prompts:
            return []

        if len(prompts) == 1 or self.max_concurrent <= 1:
            # Sequential generation
            return [self.generate(p, **kwargs) for p in prompts]

        # Parallel with thread pool (limited benefit for CPU-bound work)
        results = [None] * len(prompts)

        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            future_to_idx = {
                executor.submit(self.generate, prompt, **kwargs): idx
                for idx, prompt in enumerate(prompts)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Error generating prompt {idx}: {e}")
                    results[idx] = f"[ERROR] {e}"

        return results

    def close(self) -> None:
        """Release llama.cpp resources."""
        if hasattr(self, '_llm'):
            # llama-cpp-python Llama has a close method in newer versions
            if hasattr(self._llm, 'close'):
                self._llm.close()
            del self._llm
        logger.debug("LlamaCppLocalBackend closed")
