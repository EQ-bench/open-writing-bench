# utils/inference/http.py

"""
HTTP backend for OpenAI-compatible APIs.

Works with:
- OpenAI API
- vLLM server (--served-model-name)
- llama.cpp server (llama-server)
- Any OpenAI-compatible endpoint
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter

from .base import InferenceBackend

logger = logging.getLogger(__name__)


class HTTPBackend(InferenceBackend):
    """
    OpenAI-compatible HTTP backend using requests.

    Parallelism: Uses ThreadPoolExecutor for concurrent requests in generate_many.
    """

    KNOWN_INIT_PARAMS = {
        "model_name", "base_url", "api_key", "system_prompt",
        "timeout", "max_retries", "retry_delay", "max_concurrent"
    }

    # Common generation params that map directly to OpenAI API
    KNOWN_GEN_PARAMS = {
        "temperature", "max_tokens", "top_p", "stop",
        "presence_penalty", "frequency_penalty", "logit_bias",
        "n", "stream", "user",
        # Extended params supported by vLLM/llamacpp
        "top_k", "min_p", "repetition_penalty",
    }

    # Parameters to exclude for specific API providers (by URL substring)
    EXCLUDED_PARAMS_BY_PROVIDER = {
        "generativelanguage.googleapis.com": {"min_p", "top_k", "repetition_penalty"},
    }

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        timeout: int = 240,
        max_retries: int = 3,
        retry_delay: int = 5,
        max_concurrent: int = 64,
        **kwargs
    ):
        """
        Initialize HTTP backend.

        Args:
            model_name: Model name to send in API requests
            base_url: Base URL for the API (e.g., "https://api.openai.com/v1/chat/completions")
            api_key: API key for authentication (can also use env var)
            system_prompt: Optional system prompt to prepend to all requests
            timeout: Request timeout in seconds
            max_retries: Number of retries on failure
            retry_delay: Base delay between retries (exponential backoff)
            max_concurrent: Max concurrent requests for generate_many
            **kwargs: Additional params (will warn if unknown)
        """
        super().__init__(model_name, **kwargs)

        self.base_url = base_url
        self.api_key = api_key or os.getenv("API_KEY", "")
        self.system_prompt = system_prompt
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_concurrent = max_concurrent

        self.headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

        # Session for connection pooling - size pool to match max_concurrent
        self._session = requests.Session()
        adapter = HTTPAdapter(pool_connections=max_concurrent, pool_maxsize=max_concurrent)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)
        self._session.headers.update(self.headers)

        logger.info(f"HTTPBackend initialized: model={model_name}, url={base_url}")

    def _get_excluded_params(self) -> set[str]:
        """Get parameters to exclude based on the API provider."""
        for url_pattern, excluded in self.EXCLUDED_PARAMS_BY_PROVIDER.items():
            if url_pattern in self.base_url:
                return excluded
        return set()

    def _build_payload(self, prompt: str, **kwargs) -> dict[str, Any]:
        """Build the request payload."""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
        }

        # Get params to exclude for this provider
        excluded_params = self._get_excluded_params()

        # Add known generation params (excluding provider-specific unsupported ones)
        for param in self.KNOWN_GEN_PARAMS:
            if param in excluded_params:
                continue
            if param in kwargs and kwargs[param] is not None:
                payload[param] = kwargs[param]

        # Warn about unknown params
        unknown = set(kwargs.keys()) - self.KNOWN_GEN_PARAMS
        if unknown:
            logger.debug(f"HTTPBackend.generate: passing through unknown params: {unknown}")
            for param in unknown:
                if param in excluded_params:
                    continue
                if kwargs[param] is not None:
                    payload[param] = kwargs[param]

        # OpenRouter: disable reasoning for supported providers
        if "openrouter.ai/api/v1" in self.base_url:
            if self.model_name.startswith("anthropic/"):
                print('anthropic no reasoning')
                payload["reasoning"] = {"max_tokens": 0}
            elif self.model_name.startswith("openai/"):
                print('openai no reasoning')
                payload["reasoning"] = {"effort": "minimal"}
            if self.model_name.startswith("google/gemini-3"):
                print('gemini 3 no reasoning')
                payload["reasoning"] = {"max_tokens": 1}

        return payload

    def _make_request(self, payload: dict[str, Any], return_usage: bool = False) -> str | tuple[str, dict[str, Any] | None]:
        """Make a single request with retries.

        Args:
            payload: The request payload
            return_usage: If True, return (content, usage_dict) tuple instead of just content

        Returns:
            If return_usage is False: content string
            If return_usage is True: (content, usage_dict) tuple where usage_dict contains
                token counts and cost info (e.g. prompt_tokens, completion_tokens, total_tokens, cost)
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = self._session.post(
                    self.base_url,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]

                if return_usage:
                    usage = data.get("usage")
                    return content.strip(), usage
                return content.strip()

            except requests.exceptions.Timeout:
                logger.warning(
                    f"Request timed out (attempt {attempt + 1}/{self.max_retries})"
                )
                last_error = "Timeout"

            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response else "unknown"
                logger.warning(f"HTTP {status} error (attempt {attempt + 1}/{self.max_retries}): {e}")

                if e.response is not None:
                    try:
                        error_body = e.response.text[:500]
                        logger.debug(f"Response body: {error_body}")
                    except Exception:
                        pass

                if status == 429:
                    # Rate limit: longer backoff
                    time.sleep(self.retry_delay * (attempt + 2))
                    last_error = "Rate limited"
                    continue
                elif status in (500, 502, 503, 504):
                    # Server error: retry
                    last_error = f"HTTP {status}"
                else:
                    # Client error: don't retry
                    raise RuntimeError(f"HTTP {status} error: {e}") from e

            except requests.exceptions.RequestException as e:
                logger.warning(f"Request error (attempt {attempt + 1}/{self.max_retries}): {e}")
                last_error = str(e)

            # Exponential backoff
            time.sleep(self.retry_delay * (attempt + 1))

        error = RuntimeError(
            f"Failed to generate after {self.max_retries} attempts. Last error: {last_error}"
        )
        if return_usage:
            raise error
        raise error

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a single prompt."""
        payload = self._build_payload(prompt, **kwargs)
        return self._make_request(payload)

    def generate_with_usage(self, prompt: str, **kwargs) -> tuple[str, dict[str, Any] | None]:
        """Generate text from a single prompt and return usage info.

        Returns:
            Tuple of (content, usage_dict) where usage_dict contains:
                - prompt_tokens: int
                - completion_tokens: int
                - total_tokens: int
                - cost: float (USD cost for this request, from OpenRouter)
        """
        payload = self._build_payload(prompt, **kwargs)
        return self._make_request(payload, return_usage=True)

    def generate_many(self, prompts: list[str], **kwargs) -> list[str]:
        """
        Generate text from multiple prompts using concurrent requests.

        Uses ThreadPoolExecutor for parallelism. Results are returned
        in the same order as input prompts.
        """
        if not prompts:
            return []

        if len(prompts) == 1:
            return [self.generate(prompts[0], **kwargs)]

        results = [None] * len(prompts)
        errors = []

        with ThreadPoolExecutor(max_workers=min(self.max_concurrent, len(prompts))) as executor:
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
                    errors.append((idx, e))

        if errors:
            logger.warning(f"generate_many completed with {len(errors)} errors")

        return results

    def close(self) -> None:
        """Close the requests session."""
        self._session.close()
        logger.debug("HTTPBackend session closed")
