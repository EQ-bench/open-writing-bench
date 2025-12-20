# utils/inference/vllm_server.py

"""
vLLM OpenAI-compatible server backend - manages vllm serve process and communicates via HTTP.

This backend starts and manages a vLLM server process, providing:
- Automatic server lifecycle management (start/stop)
- OpenAI-compatible API for parallel HTTP requests
- Environment variable and CLI argument passthrough
- Health checking and readiness waiting

Unlike vllm_local.py which uses in-process inference, this backend:
- Launches vLLM as a separate server process
- Enables true request parallelism via concurrent HTTP calls
- Better utilizes vLLM's continuous batching with multiple concurrent clients
"""

import atexit
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

import requests

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


def _find_vllm_executable() -> Optional[str]:
    """
    Find the vllm executable (for 'vllm serve' command).

    Returns the path to python/vllm or None if not found.
    vLLM is typically invoked as 'vllm serve' or 'python -m vllm.entrypoints.openai.api_server'.
    """
    # Check if vllm command is available
    vllm_path = shutil.which("vllm")
    if vllm_path:
        return vllm_path

    # Fallback: check if python -m vllm works
    python_path = shutil.which("python") or shutil.which("python3")
    if python_path:
        try:
            result = subprocess.run(
                [python_path, "-c", "import vllm; print('ok')"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return python_path
        except Exception:
            pass

    return None


class VLLMServerBackend(InferenceBackend):
    """
    Backend that manages a vLLM OpenAI-compatible server and communicates via HTTP.

    This enables true request parallelism through concurrent HTTP requests,
    which works better with vLLM's continuous batching than sequential in-process calls.
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
    }

    KNOWN_INIT_PARAMS = {
        "model_name",
        # Environment variables (set before vLLM launch)
        "ENV_VARS",
        # Server configuration
        "host", "port", "timeout",
        # Sandbox configuration
        "run_sandboxed", "sandbox_user",
        # vLLM engine args (passed to vllm serve)
        "tensor_parallel_size", "pipeline_parallel_size",
        "gpu_memory_utilization", "max_model_len",
        "dtype", "quantization",
        "tokenizer", "tokenizer_mode",
        "revision", "download_dir", "seed",
        "enforce_eager", "max_num_seqs", "max_num_batched_tokens",
        "enable_prefix_caching", "disable_log_stats",
        "served_model_name",
        # HTTP client
        "max_concurrent", "max_retries", "retry_delay",
        # Startup
        "startup_timeout", "health_check_interval",
        # Extra CLI args
        "extra_args",
    }

    # Hardcoded sandbox paths
    SANDBOX_VENV_BIN = "/workspace/mounted/venvs/owl/bin"
    SANDBOX_HOME_BASE = "/workspace/mounted/vllm-sandbox"

    KNOWN_GEN_PARAMS = {
        "temperature", "max_tokens", "top_p", "top_k",
        "min_p", "repetition_penalty", "stop",
        "presence_penalty", "frequency_penalty",
    }

    # Track all active instances for cleanup
    _active_instances: list["VLLMServerBackend"] = []
    _cleanup_registered = False

    DEFAULT_PORT = 8100
    PORT_RETRY_RANGE = 10

    def __init__(
        self,
        model_name: str,
        host: str = "127.0.0.1",
        port: Optional[int] = None,
        tensor_parallel_size: Optional[int] = None,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        seed: Optional[int] = None,
        served_model_name: Optional[str] = None,
        timeout: int = 240,
        max_concurrent: int = 8,
        max_retries: int = 3,
        retry_delay: int = 5,
        startup_timeout: int = 1500,
        health_check_interval: float = 1.0,
        extra_args: Optional[list[str]] = None,
        run_sandboxed: bool = True,
        sandbox_user: str = "vllm-sandbox",
        **kwargs
    ):
        """
        Initialize vLLM server backend.

        Args:
            model_name: HuggingFace model name or path
            host: Host to bind server to
            port: Port to bind server to (default: 8100, with automatic fallback)
            tensor_parallel_size: Number of GPUs for tensor parallelism (None = all available)
            gpu_memory_utilization: Fraction of GPU memory to use
            max_model_len: Maximum sequence length (None = auto)
            dtype: Model dtype ("auto", "float16", "bfloat16", "float32")
            quantization: Quantization method (None, "awq", "gptq", etc.)
            seed: Random seed for reproducibility
            served_model_name: Name to use in API requests (defaults to model_name)
            timeout: HTTP request timeout in seconds
            max_concurrent: Max concurrent HTTP requests for generate_many
            max_retries: Number of retries on HTTP failure
            retry_delay: Base delay between retries
            startup_timeout: Max seconds to wait for server startup
            health_check_interval: Seconds between health checks during startup
            extra_args: Additional CLI arguments to pass to vllm serve
            run_sandboxed: Run vLLM in a sandboxed environment (default: True)
            sandbox_user: User to run sandboxed vLLM as (default: "vllm-sandbox")
            **kwargs: Additional vLLM engine args. Special keys:
                ENV_VARS: dict of environment variables to set before launching vLLM.
        """
        # Extract ENV_VARS before parent init
        self._env_vars = kwargs.pop("ENV_VARS", None)
        super().__init__(model_name, **kwargs)

        # Sandbox configuration
        self._run_sandboxed = run_sandboxed
        self._sandbox_user = sandbox_user

        # Find vllm executable (only needed for non-sandboxed mode)
        if not run_sandboxed:
            self._vllm_executable = _find_vllm_executable()
            if not self._vllm_executable:
                raise ImportError(
                    "vLLM is not installed or not found in PATH. "
                    "Install with: pip install vllm"
                )
        else:
            # In sandboxed mode, we use the vllm from the sandbox venv
            self._vllm_executable = f"{self.SANDBOX_VENV_BIN}/vllm"

        self.host = host
        self._requested_port = port if port is not None else self.DEFAULT_PORT
        self.port = self._requested_port
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.startup_timeout = startup_timeout
        self.health_check_interval = health_check_interval

        # Model name for API requests
        self._served_model_name = served_model_name or model_name

        # Determine trust_remote_code based on allowlist
        self._trust_remote_code = should_trust_remote_code(model_name)
        if self._trust_remote_code:
            logger.info(f"Enabling trust_remote_code for model: {model_name}")

        # Default tensor_parallel_size to number of available GPUs
        if tensor_parallel_size is None:
            tensor_parallel_size = _get_gpu_count()
            logger.info(f"Auto-detected {tensor_parallel_size} GPU(s) for tensor parallelism")

        # Store parameters for command building
        self._cmd_params = {
            "model_name": model_name,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "dtype": dtype,
            "quantization": quantization,
            "seed": seed,
            "served_model_name": self._served_model_name,
            "extra_args": extra_args or [],
            **{k: v for k, v in kwargs.items() if k in self.KNOWN_INIT_PARAMS}
        }

        # Process management
        self._process: Optional[subprocess.Popen] = None
        self._stdout_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._output_lock = threading.Lock()
        self._stdout_buffer: list[str] = []
        self._stderr_buffer: list[str] = []
        self._shutdown_event = threading.Event()

        # HTTP session
        self._session = requests.Session()

        # Register cleanup
        self._register_cleanup()

        # Start the server
        self._start_server_with_retry()

    def _build_env(self) -> dict[str, str]:
        """Build environment variables for the server process."""
        if self._run_sandboxed:
            # In sandboxed mode, use a clean environment with sandbox paths
            sandbox_home = self.SANDBOX_HOME_BASE
            venv_bin = self.SANDBOX_VENV_BIN
            env = {
                "HOME": sandbox_home,
                "HF_HOME": f"{sandbox_home}/hf",
                "HF_HUB_CACHE": f"{sandbox_home}/hf/hub",
                "TRANSFORMERS_CACHE": f"{sandbox_home}/hf/hub",
                "TMPDIR": f"{sandbox_home}/tmp",
                "PATH": f"{venv_bin}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            }
            # Add any allowed ENV_VARS from config
            if self._env_vars:
                for key, value in self._env_vars.items():
                    if self.RESTRICT_TO_ALLOWLIST and key not in self.ALLOWED_ENV_VARS:
                        logger.warning(
                            f"VLLMServerBackend: Ignoring disallowed env var in sandbox: {key}"
                        )
                        continue
                    env[key] = str(value)
            return env

        # Non-sandboxed mode: inherit environment
        env = os.environ.copy()
        if self._env_vars:
            for key, value in self._env_vars.items():
                if self.RESTRICT_TO_ALLOWLIST and key not in self.ALLOWED_ENV_VARS:
                    logger.warning(
                        f"VLLMServerBackend: Ignoring disallowed env var: {key}. "
                        f"Allowed: {self.ALLOWED_ENV_VARS}"
                    )
                    continue
                logger.debug(f"Setting env var: {key}={value}")
                env[key] = str(value)

        return env

    def _build_vllm_args(
        self,
        model_name: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_model_len: Optional[int],
        dtype: str,
        quantization: Optional[str],
        seed: Optional[int],
        served_model_name: str,
        extra_args: list[str],
        **kwargs
    ) -> list[str]:
        """Build the vllm serve arguments (without the vllm executable itself)."""
        args = ["serve", model_name]

        # Server binding
        args.extend(["--host", self.host])
        args.extend(["--port", str(self.port)])

        # Security: Control trust_remote_code based on allowlist
        if self._trust_remote_code:
            args.extend(["--trust-remote-code"])
        args.extend(["--load-format", "safetensors"])

        # Core engine args
        args.extend(["--tensor-parallel-size", str(tensor_parallel_size)])
        args.extend(["--gpu-memory-utilization", str(gpu_memory_utilization)])
        args.extend(["--dtype", dtype])

        if max_model_len is not None:
            args.extend(["--max-model-len", str(max_model_len)])

        if quantization is not None:
            args.extend(["--quantization", quantization])

        if seed is not None:
            args.extend(["--seed", str(seed)])

        if served_model_name:
            args.extend(["--served-model-name", served_model_name])

        # Optional engine args
        optional_args = {
            "pipeline_parallel_size": "--pipeline-parallel-size",
            "tokenizer": "--tokenizer",
            "tokenizer_mode": "--tokenizer-mode",
            "revision": "--revision",
            "download_dir": "--download-dir",
            "max_num_seqs": "--max-num-seqs",
            "max_num_batched_tokens": "--max-num-batched-tokens",
        }

        for param, flag in optional_args.items():
            if param in kwargs and kwargs[param] is not None:
                args.extend([flag, str(kwargs[param])])

        # Boolean flags
        bool_flags = {
            "enforce_eager": "--enforce-eager",
            "enable_prefix_caching": "--enable-prefix-caching",
            "disable_log_stats": "--disable-log-stats",
        }

        for param, flag in bool_flags.items():
            if kwargs.get(param):
                args.append(flag)

        # Suppress verbose prompt/output logging
        args.append("--disable-log-requests")
        args.append("--max-log-len")
        args.append("0")

        # Extra CLI args (passed through directly)
        if extra_args:
            args.extend(extra_args)

        return args

    def _build_server_command(
        self,
        model_name: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_model_len: Optional[int],
        dtype: str,
        quantization: Optional[str],
        seed: Optional[int],
        served_model_name: str,
        extra_args: list[str],
        **kwargs
    ) -> list[str]:
        """Build the full vllm serve command line, with sandbox wrapper if enabled."""
        vllm_args = self._build_vllm_args(
            model_name=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype=dtype,
            quantization=quantization,
            seed=seed,
            served_model_name=served_model_name,
            extra_args=extra_args,
            **kwargs
        )

        if not self._run_sandboxed:
            # Direct invocation
            if self._vllm_executable and "python" not in self._vllm_executable:
                return [self._vllm_executable] + vllm_args
            else:
                # Python module invocation (convert 'serve MODEL' to '-m vllm... --model MODEL')
                return [
                    self._vllm_executable or "python",
                    "-m", "vllm.entrypoints.openai.api_server",
                    "--model", model_name,
                ] + vllm_args[2:]  # Skip 'serve' and model_name from vllm_args

        # Sandboxed invocation using setpriv + prlimit
        # Environment is passed via subprocess env parameter, cwd set to sandbox home
        cmd = [
            "setpriv",
            f"--reuid={self._sandbox_user}",
            f"--regid={self._sandbox_user}",
            "--init-groups",
            "--",
            "prlimit",
            "--core=0:0",
            "--nproc=4096:4096",
            "--nofile=1048576:1048576",
            "--",
            f"{self.SANDBOX_VENV_BIN}/vllm",
        ]
        cmd.extend(vllm_args)

        return cmd

    def _register_cleanup(self):
        """Register cleanup handlers for graceful shutdown."""
        VLLMServerBackend._active_instances.append(self)

        if not VLLMServerBackend._cleanup_registered:
            atexit.register(VLLMServerBackend._cleanup_all)

            def signal_handler(signum, frame):
                logger.info(f"Received signal {signum}, cleaning up vLLM server instances...")
                VLLMServerBackend._cleanup_all()
                sys.exit(0)

            try:
                signal.signal(signal.SIGTERM, signal_handler)
                signal.signal(signal.SIGINT, signal_handler)
            except (ValueError, OSError):
                pass

            VLLMServerBackend._cleanup_registered = True

    @classmethod
    def _cleanup_all(cls):
        """Clean up all active server instances."""
        for instance in cls._active_instances[:]:
            try:
                instance.close()
            except Exception as e:
                logger.error(f"Error cleaning up vLLM server instance: {e}")
        cls._active_instances.clear()

    def _capture_output(self, pipe, buffer: list[str], name: str):
        """Capture output from a pipe to a buffer."""
        try:
            for line in iter(pipe.readline, ''):
                if self._shutdown_event.is_set():
                    break
                line = line.rstrip('\n\r')
                if line:
                    with self._output_lock:
                        buffer.append(line)
                    logger.debug(f"[vllm-server {name}] {line}")
        except Exception as e:
            if not self._shutdown_event.is_set():
                logger.error(f"Error capturing {name}: {e}")
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    def _start_server_with_retry(self):
        """Start the server, trying alternative ports if requested port is busy."""
        last_error = None

        for port_offset in range(self.PORT_RETRY_RANGE):
            try_port = self._requested_port + port_offset
            self.port = try_port
            self.base_url = f"http://{self.host}:{self.port}"

            self._cmd = self._build_server_command(**self._cmd_params)

            try:
                self._start_server()
                if port_offset > 0:
                    logger.info(
                        f"Successfully bound to fallback port {self.port} "
                        f"(requested port {self._requested_port} was busy)"
                    )
                return
            except RuntimeError as e:
                error_msg = str(e).lower()
                if "address already in use" in error_msg or "port" in error_msg:
                    logger.warning(f"Port {try_port} is busy, trying next port...")
                    last_error = e
                    self._cleanup_process()
                    continue
                else:
                    raise

        raise RuntimeError(
            f"Failed to bind to any port in range {self._requested_port}-"
            f"{self._requested_port + self.PORT_RETRY_RANGE - 1}. "
            f"Last error: {last_error}"
        )

    def _cleanup_process(self):
        """Clean up a failed process attempt."""
        self._shutdown_event.set()
        if self._process is not None:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None
        self._shutdown_event.clear()
        with self._output_lock:
            self._stdout_buffer.clear()
            self._stderr_buffer.clear()

    def _start_server(self):
        """Start the vLLM server process and wait for it to be ready."""
        logger.info(f"Starting vLLM server: {' '.join(self._cmd)}")

        env = self._build_env()

        # In sandboxed mode, set cwd to sandbox home for proper permissions
        cwd = self.SANDBOX_HOME_BASE if self._run_sandboxed else None

        try:
            self._process = subprocess.Popen(
                self._cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env,
                cwd=cwd,
            )
        except FileNotFoundError as e:
            if self._run_sandboxed:
                raise FileNotFoundError(
                    f"Failed to start sandboxed vLLM. Ensure 'setpriv' is available and "
                    f"the sandbox user '{self._sandbox_user}' exists. Error: {e}"
                )
            raise FileNotFoundError(
                f"vLLM executable not found: {self._vllm_executable}"
            )
        except PermissionError as e:
            if self._run_sandboxed:
                raise PermissionError(
                    f"Permission denied starting sandboxed vLLM. Ensure the current user "
                    f"can run setpriv to switch to '{self._sandbox_user}'. Error: {e}"
                )
            raise PermissionError(
                f"No execute permission for: {self._vllm_executable}"
            )

        # Start output capture threads
        self._stdout_thread = threading.Thread(
            target=self._capture_output,
            args=(self._process.stdout, self._stdout_buffer, "stdout"),
            daemon=True
        )
        self._stderr_thread = threading.Thread(
            target=self._capture_output,
            args=(self._process.stderr, self._stderr_buffer, "stderr"),
            daemon=True
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

        # Wait for server to be ready
        self._wait_for_ready()

        logger.info(f"vLLM server ready at {self.base_url}")

    def _wait_for_ready(self):
        """Wait for the server to become ready via health check."""
        health_url = f"{self.base_url}/health"
        start_time = time.time()
        last_error = None

        while time.time() - start_time < self.startup_timeout:
            # Check if process died
            if self._process.poll() is not None:
                exit_code = self._process.returncode
                stderr_output = self._get_recent_stderr()
                raise RuntimeError(
                    f"vLLM server exited unexpectedly with code {exit_code}. "
                    f"Recent stderr:\n{stderr_output}"
                )

            try:
                response = self._session.get(health_url, timeout=5)
                if response.status_code == 200:
                    return
                last_error = f"Health check returned status {response.status_code}"
            except requests.exceptions.ConnectionError:
                last_error = "Connection refused (server still starting)"
            except requests.exceptions.Timeout:
                last_error = "Health check timed out"
            except Exception as e:
                last_error = str(e)

            time.sleep(self.health_check_interval)

        stderr_output = self._get_recent_stderr()
        raise TimeoutError(
            f"vLLM server did not become ready within {self.startup_timeout}s. "
            f"Last error: {last_error}. Recent stderr:\n{stderr_output}"
        )

    def _get_recent_stderr(self, lines: int = 50) -> str:
        """Get recent stderr output for diagnostics."""
        with self._output_lock:
            recent = self._stderr_buffer[-lines:] if self._stderr_buffer else []
        return "\n".join(recent) if recent else "(no stderr output captured)"

    def _get_recent_stdout(self, lines: int = 50) -> str:
        """Get recent stdout output for diagnostics."""
        with self._output_lock:
            recent = self._stdout_buffer[-lines:] if self._stdout_buffer else []
        return "\n".join(recent) if recent else "(no stdout output captured)"

    def _build_payload(self, prompt: str, **kwargs) -> dict[str, Any]:
        """Build the request payload for chat completions endpoint."""
        messages = [{"role": "user", "content": prompt}]

        payload = {
            "model": self._served_model_name,
            "messages": messages,
        }

        # Map generation params
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
                payload[dst] = kwargs[src]

        return payload

    def _make_request(self, payload: dict[str, Any]) -> str:
        """Make a single request with retries."""
        url = f"{self.base_url}/v1/chat/completions"
        last_error = None

        for attempt in range(self.max_retries):
            # Check if server is still running
            if self._process and self._process.poll() is not None:
                stderr_output = self._get_recent_stderr()
                raise RuntimeError(
                    f"vLLM server died during request. "
                    f"Exit code: {self._process.returncode}. "
                    f"Recent stderr:\n{stderr_output}"
                )

            try:
                response = self._session.post(
                    url,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                return content.strip()

            except requests.exceptions.Timeout:
                logger.warning(
                    f"Request timed out (attempt {attempt + 1}/{self.max_retries})"
                )
                last_error = "Timeout"

            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response else "unknown"
                logger.warning(
                    f"HTTP {status} error (attempt {attempt + 1}/{self.max_retries}): {e}"
                )

                if e.response is not None:
                    try:
                        error_body = e.response.text[:500]
                        logger.debug(f"Response body: {error_body}")
                    except Exception:
                        pass

                if status == 429:
                    time.sleep(self.retry_delay * (attempt + 2))
                    last_error = "Rate limited"
                    continue
                elif status in (500, 502, 503, 504):
                    last_error = f"HTTP {status}"
                else:
                    raise RuntimeError(f"HTTP {status} error: {e}") from e

            except requests.exceptions.RequestException as e:
                logger.warning(
                    f"Request error (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                last_error = str(e)

            time.sleep(self.retry_delay * (attempt + 1))

        raise RuntimeError(
            f"Failed after {self.max_retries} attempts. Last error: {last_error}"
        )

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a single prompt."""
        payload = self._build_payload(prompt, **kwargs)
        return self._make_request(payload)

    def generate_many(self, prompts: list[str], **kwargs) -> list[str]:
        """
        Generate text from multiple prompts using concurrent HTTP requests.

        This enables true parallelism by sending multiple requests to the vLLM
        server concurrently, which utilizes vLLM's continuous batching effectively.
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

    def get_server_output(self) -> dict[str, list[str]]:
        """Get captured server output for debugging."""
        with self._output_lock:
            return {
                "stdout": self._stdout_buffer.copy(),
                "stderr": self._stderr_buffer.copy(),
            }

    def is_running(self) -> bool:
        """Check if the server process is still running."""
        return self._process is not None and self._process.poll() is None

    def close(self) -> None:
        """Stop the server and clean up resources."""
        self._shutdown_event.set()

        if self._process is not None:
            logger.info("Stopping vLLM server...")

            try:
                self._process.terminate()
                try:
                    self._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning("Server didn't stop gracefully, forcing...")
                    self._process.kill()
                    self._process.wait(timeout=5)
            except Exception as e:
                logger.error(f"Error stopping server: {e}")

            self._process = None

        for thread in [self._stdout_thread, self._stderr_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=2)

        self._session.close()

        if self in VLLMServerBackend._active_instances:
            VLLMServerBackend._active_instances.remove(self)

        logger.debug("VLLMServerBackend closed")

    def __del__(self):
        """Ensure cleanup on garbage collection."""
        try:
            if hasattr(self, '_process') and self._process is not None:
                self.close()
        except Exception:
            pass
