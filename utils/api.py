# utils/api.py

"""
Provides a provider-based abstraction for interacting with LLMs.

- `get_client`: A factory function that returns the correct client for a given
  model name and type ('test' or 'judge').
- Test models are configured in `models.yaml` and can use 'openai', 'vllm', or 'transformers' providers.
- Judge models are configured via a layered system (`config/judge_model_creds.yaml`
  overriding the database) and are restricted to OpenAI-compatible endpoints.
"""

import os
import time
import logging
import json
import requests
import yaml
import threading
import asyncio
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from inspect_ai.model import get_model, GenerateConfig, Model

from .config_loader import get_judge_config

# Conditional import of inspect-ai for local model support
try:
    import inspect_ai
    from inspect_ai.model import get_model, Model
    INSPECT_AI_AVAILABLE = True
    logging.info("inspect-ai library found and imported.")
except ImportError:
    INSPECT_AI_AVAILABLE = False
    Model = None # Define for type hinting
    logging.warning("inspect-ai library not found. vLLM and Transformers providers will be unavailable.")

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

class OpenAICompatibleClient(LLMClient):
    """Client for OpenAI-compatible API endpoints, with support for custom system prompts."""
    def __init__(self, model_name: str, api_key: str, base_url: str, system_prompt: Optional[str] = None, **kwargs):
        super().__init__(model_name, **kwargs)
        self.base_url = base_url
        self.system_prompt = system_prompt
        self.request_timeout = int(os.getenv("REQUEST_TIMEOUT", 240))
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        logging.debug(f"Initialized OpenAICompatibleClient for {model_name} at {base_url}")

    def generate(self, prompt: str, temperature: float, max_tokens: int, **kwargs) -> str:
        messages = [{"role": "user", "content": prompt}]
        # Prepend the instance's system prompt if it exists
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        for attempt in range(self.max_retries):
            response_obj = None
            try:
                payload = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                if 'min_p' in kwargs:
                    payload['min_p'] = kwargs['min_p']

                response_obj = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.request_timeout
                )
                response_obj.raise_for_status()
                data = response_obj.json()
                content = data["choices"][0]["message"]["content"]
                return content.strip()
            except requests.exceptions.Timeout:
                logging.warning(f"Request timed out for {self.model_name} on attempt {attempt+1}/{self.max_retries}")
            except requests.exceptions.HTTPError as e:
                logging.error(f"HTTPError for {self.model_name}: {e}")
                if response_obj is not None:
                    logging.error(f"Response body: {response_obj.text}")
                if e.response.status_code == 429:
                    logging.warning("Rate limit exceeded. Backing off.")
                    time.sleep(self.retry_delay * (attempt + 2)) # Longer backoff for rate limits
                    continue
            except Exception as e:
                logging.error(f"Error on attempt {attempt+1}/{self.max_retries} for {self.model_name}: {e}", exc_info=True)

            time.sleep(self.retry_delay * (attempt + 1))

        raise RuntimeError(f"Failed to generate text from {self.model_name} after {self.max_retries} attempts")



class InspectAIClient(LLMClient):
    """
    Thread-safe, singleton-cached inspect-ai client.
    - Normalizes providers: {'transformers','hf'} -> 'hf'; 'vllm' -> 'vllm'
    - Caches Model per (provider, model_name, vllm_params_file)
    - Runs all async work on one dedicated event loop thread to avoid cross-thread loop binding
    """
    _model_cache: Dict[tuple, Model] = {}
    _cache_lock = threading.Lock()

    # one process-wide loop thread
    _loop: Optional[asyncio.AbstractEventLoop] = None
    _loop_thread: Optional[threading.Thread] = None
    _loop_lock = threading.Lock()

    @classmethod
    def _ensure_loop(cls) -> asyncio.AbstractEventLoop:
        with cls._loop_lock:
            if cls._loop and cls._loop.is_running():
                return cls._loop

            def _loop_runner():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                cls._loop = loop
                loop.run_forever()

            t = threading.Thread(target=_loop_runner, name="inspect-ai-loop", daemon=True)
            t.start()
            cls._loop_thread = t

            # wait until loop is ready
            while cls._loop is None or not cls._loop.is_running():
                pass
            return cls._loop  # type: ignore

    def __init__(self, model_name: str, provider: str, vllm_params_file: Optional[str] = None,
                 max_connections: Optional[int] = None,
                 default_temperature: float = 0.7,
                 **kwargs):
        if not INSPECT_AI_AVAILABLE:
            raise RuntimeError("inspect-ai is not installed. Cannot use vLLM or Transformers providers.")
        super().__init__(model_name, **kwargs)

        prov = provider.lower().strip()
        if prov in ("transformers", "hf"):
            self.provider = "hf"
        elif prov == "vllm":
            self.provider = "vllm"
        else:
            raise ValueError(f"Unsupported inspect-ai provider '{provider}'. Use 'vllm' or 'transformers/hf'.")

        self.vllm_params_file = vllm_params_file
        self.model: Optional[Model] = None
        self.max_connections = max_connections
        self.default_temperature = default_temperature
        self.model_name = model_name

    def _resolve_or_launch_model(self) -> Model:
        cache_key = (self.provider, self.model_name, self.vllm_params_file)

        with self._cache_lock:
            cached = self._model_cache.get(cache_key)
            if cached is not None:
                self.model = cached
                return cached

        # set vLLM server args before first load
        if self.provider == "vllm" and self.vllm_params_file:
            try:
                if os.path.exists(self.vllm_params_file):
                    os.environ["VLLM_DEFAULT_SERVER_ARGS"] = json.dumps({"config": self.vllm_params_file})
                    logging.info(f"VLLM_DEFAULT_SERVER_ARGS -> {self.vllm_params_file}")
                else:
                    logging.warning(f"vLLM params file not found: {self.vllm_params_file}. Using inspect-ai defaults.")
            except Exception as e:
                logging.warning(f"Failed setting VLLM_DEFAULT_SERVER_ARGS: {e}")

        model_str = f"{self.provider}/{self.model_name}"
        gen_cfg = GenerateConfig(max_connections=self.max_connections) if self.max_connections else GenerateConfig()

        with self._cache_lock:
            cached = self._model_cache.get(cache_key)
            if cached is None:
                logging.info(f"Loading inspect-ai model: {model_str}")
                m = get_model(model_str, config=gen_cfg)
                self._model_cache[cache_key] = m
                cached = m
            self.model = cached
            return cached

        # --- unified extractor ----------------------------------------------------
    @staticmethod
    def _extract_text_any(resp) -> str:
        """
        Return best-effort text from inspect-ai ModelOutput across versions.
        Supports:
          - resp.output.completion / resp.output.text
          - resp.text / resp.completion
          - resp.output[0].text / completions[0].text
          - pydantic v2: resp.model_dump()
          - openai-like dict: choices[0].message.content / choices[0].text
        Raises RuntimeError if nothing usable is found.
        """
        # direct attrs first
        for path in [
            ("output", "completion"),
            ("output", "text"),
            ("text",),
            ("completion",),
        ]:
            try:
                cur = resp
                for p in path:
                    cur = getattr(cur, p)
                if isinstance(cur, str) and cur.strip():
                    return cur.strip()
            except Exception:
                pass

        # indexable output variants
        try:
            out = getattr(resp, "output", None)
            if isinstance(out, (list, tuple)) and out:
                # object with .text
                t = getattr(out[0], "text", None)
                if isinstance(t, str) and t.strip():
                    return t.strip()
        except Exception:
            pass

        # pydantic v2 (model_dump)
        try:
            dump = None
            if hasattr(resp, "model_dump"):
                dump = resp.model_dump()
            elif hasattr(resp, "dict"):
                dump = resp.dict()
            if isinstance(dump, dict) and dump:
                # inspect-ai canonical dumps sometimes have {"output": {"text": "..."}}
                cand = (
                    (((dump.get("output") or {}).get("completion")) or (dump.get("output") or {}).get("text"))
                    or dump.get("text")
                )
                if isinstance(cand, str) and cand.strip():
                    return cand.strip()
                # openai-like
                ch = dump.get("choices")
                if isinstance(ch, list) and ch:
                    msg = (ch[0].get("message") or {})
                    if isinstance(msg, dict) and msg.get("content"):
                        return str(msg["content"]).strip()
                    if ch[0].get("text"):
                        return str(ch[0]["text"]).strip()
        except Exception:
            pass

        # raw dict fallback
        if isinstance(resp, dict):
            cand = (
                (((resp.get("output") or {}).get("completion")) or (resp.get("output") or {}).get("text"))
                or resp.get("text")
            )
            if isinstance(cand, str) and cand.strip():
                return cand.strip()
            ch = resp.get("choices")
            if isinstance(ch, list) and ch:
                msg = (ch[0].get("message") or {})
                if isinstance(msg, dict) and msg.get("content"):
                    return str(msg["content"]).strip()
                if ch[0].get("text"):
                    return str(ch[0]["text"]).strip()

        raise RuntimeError("inspect-ai: could not extract text from ModelOutput")

    async def _generate_async(self, prompt: str, temperature: float, max_tokens: int, **kwargs) -> str:
        model = self._resolve_or_launch_model()
        cfg = GenerateConfig(
            temperature=temperature if temperature is not None else self.default_temperature,
            max_tokens=max_tokens,
        )
        resp = await model.generate(input=prompt, config=cfg)
        return self._extract_text_any(resp)


    def generate(self, prompt: str, temperature: float, max_tokens: int, **kwargs) -> str:
        loop = self._ensure_loop()
        fut = asyncio.run_coroutine_threadsafe(
            self._generate_async(prompt, temperature, max_tokens, **kwargs),
            loop
        )
        return fut.result()

        # --- batch API ------------------------------------------------------------
    async def _generate_many_async(self, prompts: list[str], temperature: float, max_tokens: int) -> list[str]:
        model = self._resolve_or_launch_model()
        cfg = GenerateConfig(
            temperature=temperature if temperature is not None else self.default_temperature,
            max_tokens=max_tokens,
        )
        # Try native batched call first (inspect-ai supports list input in recent versions)
        try:
            resp = await model.generate(input=prompts, config=cfg)
            # resp may be a list-like of ModelOutput or a single object with .output as list
            outs = []
            if isinstance(resp, (list, tuple)):
                for r in resp:
                    outs.append(self._extract_text_any(r))
            else:
                # try resp.output as a list
                out = getattr(resp, "output", None)
                if isinstance(out, (list, tuple)):
                    for r in out:
                        outs.append(self._extract_text_any(r))
                else:
                    # fallback: single response replicated? treat as single
                    outs.append(self._extract_text_any(resp))
            if len(outs) == len(prompts):
                return outs
        except Exception:
            # fall back to parallel per-prompt on the same loop
            tasks = [model.generate(input=p, config=cfg) for p in prompts]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            outs = []
            for r in results:
                if isinstance(r, Exception):
                    outs.append(f"[ERROR] {r}")
                else:
                    try:
                        outs.append(self._extract_text_any(r))
                    except Exception as e:
                        outs.append(f"[ERROR] {e}")
            return outs

    def generate_many(self, prompts: list[str], temperature: float, max_tokens: int) -> list[str]:
        loop = self._ensure_loop()
        fut = asyncio.run_coroutine_threadsafe(
            self._generate_many_async(prompts, temperature, max_tokens),
            loop
        )
        return fut.result()


def get_client(name_or_key: str, client_type: str,
               vllm_params_file: Optional[str] = None,
               test_provider: Optional[str] = None) -> LLMClient:
    """
    Factory function to create clients for either 'test' or 'judge' models.
    """
    if client_type == 'test':
        # If CLI provided a provider, use it; else fall back to models.yaml
        if test_provider:
            if test_provider == 'openai':
                api_key = os.getenv("TEST_API_KEY")
                base_url = os.getenv("TEST_API_URL")
                if not api_key or not base_url:
                    raise ValueError("TEST_API_KEY or TEST_API_URL is not set in environment for test provider 'openai'.")
                return OpenAICompatibleClient(name_or_key, api_key, base_url)
            elif test_provider in ('vllm', 'transformers', 'hf'):
                if not INSPECT_AI_AVAILABLE:
                    raise RuntimeError("inspect-ai is required for vllm/transformers providers.")
                return InspectAIClient(
                    name_or_key,
                    provider=('hf' if test_provider in ('transformers', 'hf') else 'vllm'),
                    vllm_params_file=vllm_params_file,
                    max_connections=(int(os.getenv("INSPECT_MAX_CONNECTIONS")) if os.getenv("INSPECT_MAX_CONNECTIONS") else None),
                )

            else:
                raise ValueError(f"Unsupported test_provider '{test_provider}'.")
        # Fallback: models.yaml configuration path (legacy)
        configs = _load_test_model_configs()
        if name_or_key not in configs:
            raise ValueError(f"Test model '{name_or_key}' not found in models.yaml configuration, "
                             "and no --test-provider was supplied.")
        config = configs[name_or_key]
        provider = config.get('provider')
        if provider == 'openai':
            api_key_env = config.get('api_key_env')
            base_url_env = config.get('base_url_env')
            if not api_key_env or not base_url_env:
                raise ValueError(f"api_key_env or base_url_env not set for test model {name_or_key}")
            api_key = os.getenv(api_key_env)
            base_url = os.getenv(base_url_env)
            if not api_key or not base_url:
                raise ValueError(f"API key or URL environment variable not set for model {name_or_key}")
            return OpenAICompatibleClient(name_or_key, api_key, base_url)
        elif provider in ['vllm', 'transformers', 'hf']:
            return InspectAIClient(
                name_or_key,
                provider=('hf' if provider in ('transformers', 'hf') else 'vllm'),
                vllm_params_file=vllm_params_file,
                max_connections=(int(os.getenv("INSPECT_MAX_CONNECTIONS")) if os.getenv("INSPECT_MAX_CONNECTIONS") else None),
            )

        else:
            raise ValueError(f"Unsupported provider '{provider}' for test model '{name_or_key}'.")


    elif client_type == 'judge':
        config = get_judge_config(name_or_key) # Use the new layered config loader
        provider = config.get('provider')

        if provider != 'openai':
            raise ValueError(f"Unsupported provider '{provider}' for judge '{name_or_key}'. Only 'openai' (OpenAI-compatible) is supported for judges.")

        return OpenAICompatibleClient(
            model_name=config['model_id'],
            api_key=config['api_key'],
            base_url=config['base_url'],
            system_prompt=config.get('system_prompt')
        )
    else:
        raise ValueError(f"Invalid client_type specified: '{client_type}'. Must be 'test' or 'judge'.")