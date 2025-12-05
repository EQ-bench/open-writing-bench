# utils/inference/transformers_local.py

"""
Local HuggingFace Transformers backend for in-process inference.

Requires: pip install transformers torch
"""

import logging
from typing import Any, Optional, Union

from .base import InferenceBackend

logger = logging.getLogger(__name__)


class TransformersLocalBackend(InferenceBackend):
    """
    In-process HuggingFace Transformers inference backend.

    Parallelism: Uses native batching with padding for generate_many.
    GPU parallelism handled by PyTorch (device_map="auto" for multi-GPU).
    """

    KNOWN_INIT_PARAMS = {
        "model_name",
        # Model loading params
        "device_map", "torch_dtype", "load_in_8bit", "load_in_4bit",
        "trust_remote_code", "revision", "token",
        "attn_implementation", "use_flash_attention_2",
        "low_cpu_mem_usage", "offload_folder",
        # Tokenizer params
        "tokenizer_name", "padding_side",
    }

    KNOWN_GEN_PARAMS = {
        "temperature", "max_tokens", "max_new_tokens", "top_p", "top_k",
        "min_p", "repetition_penalty", "do_sample",
        "num_beams", "early_stopping", "length_penalty",
        "no_repeat_ngram_size", "pad_token_id", "eos_token_id",
    }

    def __init__(
        self,
        model_name: str,
        device_map: Union[str, dict] = "auto",
        torch_dtype: Optional[str] = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        trust_remote_code: bool = False,
        attn_implementation: Optional[str] = None,
        tokenizer_name: Optional[str] = None,
        padding_side: str = "left",  # left padding for batch generation
        **kwargs
    ):
        """
        Initialize Transformers backend.

        Args:
            model_name: HuggingFace model name or path
            device_map: Device placement ("auto", "cuda", "cpu", or dict)
            torch_dtype: Model dtype ("auto", "float16", "bfloat16", "float32")
            load_in_8bit: Use 8-bit quantization (requires bitsandbytes)
            load_in_4bit: Use 4-bit quantization (requires bitsandbytes)
            trust_remote_code: Trust remote code in HF models
            attn_implementation: Attention implementation ("flash_attention_2", "sdpa", etc.)
            tokenizer_name: Override tokenizer (default: same as model_name)
            padding_side: Padding side for batch generation ("left" recommended)
            **kwargs: Additional model loading params
        """
        super().__init__(model_name, **kwargs)

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "transformers and torch are not installed. Install with: "
                "pip install transformers torch"
            ) from e

        self._torch = torch

        # Resolve dtype
        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        resolved_dtype = dtype_map.get(torch_dtype, torch_dtype) if torch_dtype else "auto"

        # Build model kwargs
        model_kwargs = {
            "device_map": device_map,
            "trust_remote_code": trust_remote_code,
        }

        if resolved_dtype != "auto":
            model_kwargs["torch_dtype"] = resolved_dtype

        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        elif load_in_4bit:
            model_kwargs["load_in_4bit"] = True

        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        # Pass through additional model kwargs
        passthrough = {"revision", "token", "low_cpu_mem_usage", "offload_folder"}
        for key in passthrough:
            if key in kwargs and kwargs[key] is not None:
                model_kwargs[key] = kwargs[key]

        logger.info(f"Loading transformers model: {model_name}")
        logger.debug(f"Model kwargs: {model_kwargs}")

        # Load tokenizer
        tokenizer_id = tokenizer_name or model_name
        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id,
            trust_remote_code=trust_remote_code,
            padding_side=padding_side,
        )

        # Ensure pad token exists (required for batching)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Load model
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )

        logger.info(f"Transformers model loaded: {model_name}")

    def _build_generate_kwargs(self, **kwargs) -> dict[str, Any]:
        """Build kwargs for model.generate()."""
        gen_kwargs = {}

        # Handle max_tokens -> max_new_tokens mapping
        if "max_tokens" in kwargs and kwargs["max_tokens"] is not None:
            gen_kwargs["max_new_tokens"] = kwargs["max_tokens"]
        if "max_new_tokens" in kwargs and kwargs["max_new_tokens"] is not None:
            gen_kwargs["max_new_tokens"] = kwargs["max_new_tokens"]

        # Temperature handling
        if "temperature" in kwargs and kwargs["temperature"] is not None:
            temp = kwargs["temperature"]
            if temp > 0:
                gen_kwargs["temperature"] = temp
                gen_kwargs["do_sample"] = True
            else:
                gen_kwargs["do_sample"] = False

        # Other params
        param_mapping = {
            "top_p": "top_p",
            "top_k": "top_k",
            "repetition_penalty": "repetition_penalty",
            "do_sample": "do_sample",
            "num_beams": "num_beams",
            "early_stopping": "early_stopping",
            "length_penalty": "length_penalty",
            "no_repeat_ngram_size": "no_repeat_ngram_size",
        }

        for src, dst in param_mapping.items():
            if src in kwargs and kwargs[src] is not None:
                gen_kwargs[dst] = kwargs[src]

        # Set pad_token_id to avoid warnings
        if "pad_token_id" not in gen_kwargs:
            gen_kwargs["pad_token_id"] = self._tokenizer.pad_token_id

        # Warn about unknown params
        all_known = set(param_mapping.keys()) | {"max_tokens", "max_new_tokens", "temperature"}
        unknown = set(kwargs.keys()) - all_known
        if unknown:
            logger.debug(f"TransformersLocalBackend: ignoring unknown gen params: {unknown}")

        return gen_kwargs

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a single prompt."""
        gen_kwargs = self._build_generate_kwargs(**kwargs)

        # Tokenize
        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[1]

        # Generate
        with self._torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                **gen_kwargs
            )

        # Decode only the new tokens
        generated_ids = outputs[0][input_length:]
        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

        return text.strip()

    def generate_many(self, prompts: list[str], **kwargs) -> list[str]:
        """
        Generate text from multiple prompts using batched generation.

        Uses left-padding for proper batch generation with causal LMs.
        """
        if not prompts:
            return []

        if len(prompts) == 1:
            return [self.generate(prompts[0], **kwargs)]

        gen_kwargs = self._build_generate_kwargs(**kwargs)

        # Batch tokenize with padding
        inputs = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        # Track input lengths for each prompt (accounting for padding)
        attention_mask = inputs["attention_mask"]
        input_lengths = attention_mask.sum(dim=1).tolist()

        # Generate
        with self._torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                **gen_kwargs
            )

        # Decode each output, skipping the input tokens
        results = []
        for i, (output, input_len) in enumerate(zip(outputs, input_lengths)):
            # Find where the actual input starts (skip padding)
            generated_ids = output[input_len:]
            text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
            results.append(text.strip())

        return results

    def close(self) -> None:
        """Release model resources."""
        if hasattr(self, '_model'):
            del self._model
        if hasattr(self, '_tokenizer'):
            del self._tokenizer

        # Clear CUDA cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        logger.debug("TransformersLocalBackend closed")
