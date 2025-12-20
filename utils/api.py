# utils/api.py

"""
Provides a provider-based abstraction for interacting with LLMs.

- `get_client`: A factory function that returns the correct client for a given
  model name and type ('test' or 'judge').
- Test models are configured in `models.yaml` or via CLI provider flags.
- Judge models are configured via a layered system (`config/judge_model_creds.yaml`
  overriding the database) and are restricted to OpenAI-compatible endpoints.
"""

import os
import logging
import yaml
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from dotenv import load_dotenv

from .config_loader import get_judge_config
from .inference import get_backend, InferenceBackend
from .inference.base import strip_thinking_tags

load_dotenv()

_test_model_configs: Optional[Dict[str, Any]] = None


def _load_test_model_configs(config_file: str = 'config/models.yaml') -> Dict[str, Any]:
    """Loads and caches TEST model provider configurations from a YAML file."""
    global _test_model_configs
    if _test_model_configs is None:
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                all_configs = yaml.safe_load(f)
                _test_model_configs = {model['name']: model for model in all_configs.get('models', [])}
                logging.info(f"Loaded {len(_test_model_configs)} test model configurations from {config_file}.")
        except FileNotFoundError:
            logging.error(f"Test model configuration file not found: {config_file}")
            _test_model_configs = {}
        except Exception as e:
            logging.error(f"Error parsing test model configuration file {config_file}: {e}", exc_info=True)
            _test_model_configs = {}
    return _test_model_configs


class LLMClient(ABC):
    """Abstract base class for all LLM API clients."""

    def __init__(self, model_name: str, max_retries: int = 3, retry_delay: int = 5):
        self.model_name = model_name
        self.max_retries = int(os.getenv("MAX_RETRIES", max_retries))
        self.retry_delay = int(os.getenv("RETRY_DELAY", retry_delay))

    @abstractmethod
    def generate(self, prompt: str, temperature: float, max_tokens: int, **kwargs) -> str:
        """Generates text from a prompt."""
        pass


class InferenceBackendClient(LLMClient):
    """
    LLMClient wrapper around the new InferenceBackend system.

    Provides backward compatibility with existing code that expects LLMClient interface.
    """

    def __init__(self, backend: InferenceBackend, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)
        self._backend = backend

    def generate(self, prompt: str, temperature: float, max_tokens: int, **kwargs) -> str:
        """Generate text using the underlying backend."""
        result = self._backend.generate(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        return strip_thinking_tags(result)

    def generate_with_usage(self, prompt: str, temperature: float, max_tokens: int, **kwargs) -> tuple[str, dict | None]:
        """Generate text and return usage info (token counts and cost).

        Returns:
            Tuple of (content, usage_dict) where usage_dict contains:
                - prompt_tokens: int
                - completion_tokens: int
                - total_tokens: int
                - cost: float (USD cost for this request)
            Returns (content, None) if backend doesn't support usage tracking.
        """
        if hasattr(self._backend, 'generate_with_usage'):
            result, usage = self._backend.generate_with_usage(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return strip_thinking_tags(result), usage
        else:
            # Fallback for backends that don't support usage tracking
            result = self._backend.generate(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return strip_thinking_tags(result), None

    def generate_many(self, prompts: list[str], temperature: float, max_tokens: int, **kwargs) -> list[str]:
        """Generate text from multiple prompts using the backend's native batching."""
        results = self._backend.generate_many(
            prompts,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        return [strip_thinking_tags(r) for r in results]

    def close(self) -> None:
        """Release backend resources."""
        self._backend.close()


def _create_backend_from_config(
    model_name: str,
    provider: str,
    backend_config: Optional[Dict[str, Any]] = None
) -> InferenceBackend:
    """
    Create an inference backend based on provider and config.

    Args:
        model_name: Model name or path
        provider: Provider type (http, vllm, llamacpp, transformers, etc.)
        backend_config: Optional backend-specific configuration

    Returns:
        Configured InferenceBackend instance
    """
    config = backend_config or {}

    # Normalize provider names
    provider_lower = provider.lower().strip()
    provider_mapping = {
        "openai": "http",
        "hf": "transformers",
    }
    backend_provider = provider_mapping.get(provider_lower, provider_lower)

    # Merge in environment variables for HTTP backends
    if backend_provider == "http":
        if "base_url" not in config:
            config["base_url"] = os.getenv("TEST_API_URL", "")
        if "api_key" not in config:
            config["api_key"] = os.getenv("TEST_API_KEY", "")
        # Note: timeout is handled by the backend itself via REQUEST_TIMEOUT env var
        if "max_retries" not in config:
            config["max_retries"] = int(os.getenv("MAX_RETRIES", 3))
        if "retry_delay" not in config:
            config["retry_delay"] = int(os.getenv("RETRY_DELAY", 5))

    # Create the backend
    return get_backend(
        provider=backend_provider,
        model_name=model_name,
        **config
    )


def get_client(
    name_or_key: str,
    client_type: str,
    vllm_params_file: Optional[str] = None,
    test_provider: Optional[str] = None,
    backend_config: Optional[Dict[str, Any]] = None
) -> LLMClient:
    """
    Factory function to create clients for either 'test' or 'judge' models.

    Args:
        name_or_key: Model name (for test) or judge key (for judge)
        client_type: Either 'test' or 'judge'
        vllm_params_file: Deprecated - use backend_config instead
        test_provider: Provider for test models (http, vllm, llamacpp, transformers)
        backend_config: Backend-specific configuration dict

    Returns:
        LLMClient instance wrapping the appropriate backend
    """
    if client_type == 'test':
        config = backend_config.copy() if backend_config else {}

        # Handle legacy vllm_params_file
        if vllm_params_file and "vllm_params_file" not in config:
            config["vllm_params_file"] = vllm_params_file
            logging.warning("vllm_params_file is deprecated. Use backend_config instead.")

        # Determine provider
        if test_provider:
            provider = test_provider
        else:
            # Fallback to models.yaml configuration
            configs = _load_test_model_configs()
            if name_or_key in configs:
                model_config = configs[name_or_key]
                provider = model_config.get('provider', 'http')

                # Merge YAML config into backend config
                for key, value in model_config.items():
                    if key not in ('name', 'provider') and key not in config:
                        # Handle env var references
                        if isinstance(value, str) and value.endswith('_env'):
                            actual_key = key.replace('_env', '')
                            config[actual_key] = os.getenv(value.replace('_env', '').upper())
                        elif key == 'api_key_env':
                            config['api_key'] = os.getenv(value, '')
                        elif key == 'base_url_env':
                            config['base_url'] = os.getenv(value, '')
                        else:
                            config[key] = value
            else:
                raise ValueError(
                    f"Test model '{name_or_key}' not found in models.yaml configuration, "
                    "and no --test-provider was supplied."
                )

        # Create backend and wrap in client
        backend = _create_backend_from_config(name_or_key, provider, config)
        return InferenceBackendClient(backend, name_or_key)

    elif client_type == 'judge':
        # Judge models always use HTTP backend
        judge_config = get_judge_config(name_or_key)

        if judge_config.get('provider') != 'openai':
            raise ValueError(
                f"Unsupported provider '{judge_config.get('provider')}' for judge '{name_or_key}'. "
                "Only 'openai' (OpenAI-compatible) is supported for judges."
            )

        # Get judge-specific retry settings from env vars
        max_retries = int(os.getenv("MAX_JUDGE_RETRIES", os.getenv("MAX_RETRIES", 3)))
        retry_delay = int(os.getenv("JUDGE_RETRY_DELAY", os.getenv("RETRY_DELAY", 5)))

        backend = get_backend(
            provider="http",
            model_name=judge_config['model_id'],
            base_url=judge_config['base_url'],
            api_key=judge_config['api_key'],
            system_prompt=judge_config.get('system_prompt'),
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        return InferenceBackendClient(backend, judge_config['model_id'])

    else:
        raise ValueError(f"Invalid client_type specified: '{client_type}'. Must be 'test' or 'judge'.")
