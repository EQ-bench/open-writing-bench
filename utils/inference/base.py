# utils/inference/base.py

"""
Base classes and utilities for inference backends.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """
    Configuration for inference requests.

    Common parameters are defined explicitly; backend-specific params
    go in `extra` and are passed through to the backend.
    """
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    stop: Optional[list[str]] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_none: bool = False) -> dict[str, Any]:
        """Convert to dict, optionally excluding None values."""
        result = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        optional_fields = ["top_p", "top_k", "min_p", "repetition_penalty", "stop"]
        for f in optional_fields:
            val = getattr(self, f)
            if val is not None or include_none:
                result[f] = val
        result.update(self.extra)
        return result

    @classmethod
    def from_kwargs(cls, **kwargs) -> "InferenceConfig":
        """
        Create config from kwargs, separating known fields from extras.
        """
        known_fields = {"temperature", "max_tokens", "top_p", "top_k",
                        "min_p", "repetition_penalty", "stop"}
        known = {k: v for k, v in kwargs.items() if k in known_fields}
        extra = {k: v for k, v in kwargs.items() if k not in known_fields}
        return cls(**known, extra=extra)


class InferenceBackend(ABC):
    """
    Abstract base class for all inference backends.

    Subclasses must implement:
        - generate(prompt, **kwargs) -> str
        - close() -> None

    Subclasses may override:
        - generate_many(prompts, **kwargs) -> list[str] for native batching
    """

    # Known init params for this backend (for validation/warning)
    KNOWN_INIT_PARAMS: set[str] = {"model_name"}

    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the backend.

        Args:
            model_name: The model identifier (HF repo, path, or API model name)
            **kwargs: Backend-specific configuration
        """
        self.model_name = model_name
        self._validate_init_params(kwargs)

    def _validate_init_params(self, kwargs: dict[str, Any]) -> None:
        """Warn about unknown init parameters."""
        unknown = set(kwargs.keys()) - self.KNOWN_INIT_PARAMS
        if unknown:
            logger.warning(
                f"{self.__class__.__name__}: Unknown init params (will be ignored): {unknown}"
            )

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from a single prompt.

        Args:
            prompt: The input prompt
            **kwargs: Generation parameters (temperature, max_tokens, etc.)

        Returns:
            Generated text
        """
        pass

    def generate_many(self, prompts: list[str], **kwargs) -> list[str]:
        """
        Generate text from multiple prompts.

        Default implementation: sequential generation.
        Override in subclasses for native batching or parallel execution.

        Args:
            prompts: List of input prompts
            **kwargs: Generation parameters (applied to all prompts)

        Returns:
            List of generated texts (same order as prompts)
        """
        return [self.generate(p, **kwargs) for p in prompts]

    @abstractmethod
    def close(self) -> None:
        """
        Release any resources held by the backend.

        Called when the backend is no longer needed.
        """
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
