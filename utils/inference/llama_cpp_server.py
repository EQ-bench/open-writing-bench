# utils/inference/llama_cpp_server.py

"""
llama.cpp server backend - manages llama-server process and communicates via HTTP.

This backend starts and manages a llama-server process, providing:
- Automatic server lifecycle management (start/stop)
- Output capture for error logging
- Clean shutdown handling
- Health checking and readiness waiting

The server path can be configured via:
1. LLAMA_CPP_SERVER_PATH environment variable
2. Default path: /workspace/llama.cpp/build/bin/llama-server
3. Common fallback locations if neither exists
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

logger = logging.getLogger(__name__)


def _find_llama_server() -> Optional[str]:
    """
    Find the llama-server executable.

    Search order:
    1. LLAMA_CPP_SERVER_PATH environment variable
    2. Default path: /workspace/llama.cpp/build/bin/llama-server
    3. Common installation locations
    4. System PATH
    """
    # Check environment variable first
    env_path = os.environ.get("LLAMA_CPP_SERVER_PATH")
    if env_path and os.path.isfile(env_path) and os.access(env_path, os.X_OK):
        return env_path

    # Default path
    default_path = "/workspace/llama.cpp/build/bin/llama-server"
    if os.path.isfile(default_path) and os.access(default_path, os.X_OK):
        return default_path

    # Common fallback locations
    fallback_paths = [
        os.path.expanduser("~/llama.cpp/build/bin/llama-server"),
        os.path.expanduser("~/.local/bin/llama-server"),
        "/usr/local/bin/llama-server",
        "/opt/llama.cpp/build/bin/llama-server",
        # Relative to current directory
        "./llama.cpp/build/bin/llama-server",
        "../llama.cpp/build/bin/llama-server",
    ]

    for path in fallback_paths:
        expanded = os.path.expanduser(path)
        if os.path.isfile(expanded) and os.access(expanded, os.X_OK):
            return expanded

    # Check system PATH
    which_result = shutil.which("llama-server")
    if which_result:
        return which_result

    return None


class LlamaCppServerBackend(InferenceBackend):
    """
    Backend that manages a llama-server process and communicates via HTTP.

    This is useful for:
    - Production deployments where server management is needed
    - Cases where you want llama.cpp's native HTTP server batching
    - Long-running services with proper lifecycle management
    """

    KNOWN_INIT_PARAMS = {
        "model_name",  # path to GGUF file
        # Server configuration
        "server_path", "host", "port", "timeout",
        # llama.cpp model params (passed to server)
        "n_ctx", "n_batch", "n_threads", "n_threads_batch",
        "n_gpu_layers", "main_gpu", "tensor_split",
        "use_mmap", "use_mlock", "flash_attn",
        "rope_freq_base", "rope_freq_scale",
        # Server behavior
        "n_parallel", "cont_batching",
        # HTTP client
        "max_concurrent", "max_retries", "retry_delay",
        # Startup
        "startup_timeout", "health_check_interval",
    }

    KNOWN_GEN_PARAMS = {
        "temperature", "max_tokens", "top_p", "top_k",
        "min_p", "repeat_penalty", "stop",
        "presence_penalty", "frequency_penalty",
        "mirostat", "mirostat_tau", "mirostat_eta",
        "tfs_z", "typical_p", "seed",
    }

    # Track all active instances for cleanup
    _active_instances: list["LlamaCppServerBackend"] = []
    _cleanup_registered = False

    def __init__(
        self,
        model_name: str,  # path to GGUF file
        server_path: Optional[str] = None,
        host: str = "127.0.0.1",
        port: int = 8080,
        n_ctx: int = 4096,
        n_batch: int = 512,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = -1,
        n_parallel: int = 1,
        cont_batching: bool = True,
        flash_attn: bool = False,
        timeout: int = 240,
        max_concurrent: int = 8,
        max_retries: int = 3,
        retry_delay: int = 5,
        startup_timeout: int = 120,
        health_check_interval: float = 0.5,
        **kwargs
    ):
        """
        Initialize llama.cpp server backend.

        Args:
            model_name: Path to GGUF model file
            server_path: Path to llama-server executable (auto-detected if None)
            host: Host to bind server to
            port: Port to bind server to
            n_ctx: Context window size
            n_batch: Batch size for prompt processing
            n_threads: Number of CPU threads (None = auto)
            n_gpu_layers: Layers to offload to GPU (-1 = all)
            n_parallel: Number of parallel sequences to handle
            cont_batching: Enable continuous batching
            flash_attn: Enable flash attention
            timeout: HTTP request timeout in seconds
            max_concurrent: Max concurrent HTTP requests
            max_retries: Number of retries on HTTP failure
            retry_delay: Base delay between retries
            startup_timeout: Max seconds to wait for server startup
            health_check_interval: Seconds between health checks during startup
            **kwargs: Additional llama-server args
        """
        super().__init__(model_name, **kwargs)

        # Find server executable
        self.server_path = server_path or _find_llama_server()
        if not self.server_path:
            raise FileNotFoundError(
                "llama-server not found. Set LLAMA_CPP_SERVER_PATH environment variable "
                "or install llama.cpp and build llama-server. "
                "Searched: LLAMA_CPP_SERVER_PATH env, /workspace/llama.cpp/build/bin/llama-server, "
                "common paths, and system PATH."
            )

        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.startup_timeout = startup_timeout
        self.health_check_interval = health_check_interval

        # Build server command
        self._cmd = self._build_server_command(
            model_name=model_name,
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            n_parallel=n_parallel,
            cont_batching=cont_batching,
            flash_attn=flash_attn,
            **kwargs
        )

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
        self._start_server()

    def _build_server_command(
        self,
        model_name: str,
        n_ctx: int,
        n_batch: int,
        n_threads: Optional[int],
        n_gpu_layers: int,
        n_parallel: int,
        cont_batching: bool,
        flash_attn: bool,
        **kwargs
    ) -> list[str]:
        """Build the llama-server command line."""
        cmd = [
            self.server_path,
            "--model", model_name,
            "--host", self.host,
            "--port", str(self.port),
            "--ctx-size", str(n_ctx),
            "--batch-size", str(n_batch),
            "--n-gpu-layers", str(n_gpu_layers),
            "--parallel", str(n_parallel),
        ]

        if n_threads is not None:
            cmd.extend(["--threads", str(n_threads)])

        if cont_batching:
            cmd.append("--cont-batching")

        if flash_attn:
            cmd.extend(["--flash-attn", "on"])

        # Pass through additional server args
        passthrough_args = {
            "n_threads_batch": "--threads-batch",
            "main_gpu": "--main-gpu",
            "tensor_split": "--tensor-split",
            "use_mmap": "--mmap",
            "use_mlock": "--mlock",
            "rope_freq_base": "--rope-freq-base",
            "rope_freq_scale": "--rope-freq-scale",
        }

        for param, flag in passthrough_args.items():
            if param in kwargs and kwargs[param] is not None:
                value = kwargs[param]
                if isinstance(value, bool):
                    if value:
                        cmd.append(flag)
                else:
                    cmd.extend([flag, str(value)])

        return cmd

    def _register_cleanup(self):
        """Register cleanup handlers for graceful shutdown."""
        LlamaCppServerBackend._active_instances.append(self)

        if not LlamaCppServerBackend._cleanup_registered:
            atexit.register(LlamaCppServerBackend._cleanup_all)

            # Handle SIGTERM/SIGINT gracefully
            def signal_handler(signum, frame):
                logger.info(f"Received signal {signum}, cleaning up llama-server instances...")
                LlamaCppServerBackend._cleanup_all()
                sys.exit(0)

            try:
                signal.signal(signal.SIGTERM, signal_handler)
                signal.signal(signal.SIGINT, signal_handler)
            except (ValueError, OSError):
                # Signal handling may not work in all contexts (e.g., threads)
                pass

            LlamaCppServerBackend._cleanup_registered = True

    @classmethod
    def _cleanup_all(cls):
        """Clean up all active server instances."""
        for instance in cls._active_instances[:]:
            try:
                instance.close()
            except Exception as e:
                logger.error(f"Error cleaning up server instance: {e}")
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
                    # Log server output at debug level
                    logger.debug(f"[llama-server {name}] {line}")
        except Exception as e:
            if not self._shutdown_event.is_set():
                logger.error(f"Error capturing {name}: {e}")
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    def _start_server(self):
        """Start the llama-server process and wait for it to be ready."""
        logger.info(f"Starting llama-server: {' '.join(self._cmd)}")

        try:
            self._process = subprocess.Popen(
                self._cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"llama-server executable not found at: {self.server_path}"
            )
        except PermissionError:
            raise PermissionError(
                f"No execute permission for llama-server at: {self.server_path}"
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

        logger.info(f"llama-server ready at {self.base_url}")

    def _wait_for_ready(self):
        """Wait for the server to become ready, with health checks."""
        health_url = f"{self.base_url}/health"
        start_time = time.time()
        last_error = None

        while time.time() - start_time < self.startup_timeout:
            # Check if process died
            if self._process.poll() is not None:
                exit_code = self._process.returncode
                stderr_output = self._get_recent_stderr()
                raise RuntimeError(
                    f"llama-server exited unexpectedly with code {exit_code}. "
                    f"Recent stderr:\n{stderr_output}"
                )

            try:
                response = self._session.get(health_url, timeout=5)
                if response.status_code == 200:
                    return  # Server is ready
                last_error = f"Health check returned status {response.status_code}"
            except requests.exceptions.ConnectionError:
                last_error = "Connection refused (server still starting)"
            except requests.exceptions.Timeout:
                last_error = "Health check timed out"
            except Exception as e:
                last_error = str(e)

            time.sleep(self.health_check_interval)

        # Timeout - get diagnostic info
        stderr_output = self._get_recent_stderr()
        raise TimeoutError(
            f"llama-server did not become ready within {self.startup_timeout}s. "
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
        """Build the request payload for /completion endpoint."""
        payload = {
            "prompt": prompt,
        }

        # Map generation params
        param_mapping = {
            "temperature": "temperature",
            "max_tokens": "n_predict",
            "top_p": "top_p",
            "top_k": "top_k",
            "min_p": "min_p",
            "repeat_penalty": "repeat_penalty",
            "repetition_penalty": "repeat_penalty",  # alias
            "stop": "stop",
            "presence_penalty": "presence_penalty",
            "frequency_penalty": "frequency_penalty",
            "mirostat": "mirostat",
            "mirostat_tau": "mirostat_tau",
            "mirostat_eta": "mirostat_eta",
            "tfs_z": "tfs_z",
            "typical_p": "typical_p",
            "seed": "seed",
        }

        for src, dst in param_mapping.items():
            if src in kwargs and kwargs[src] is not None:
                payload[dst] = kwargs[src]

        return payload

    def _make_request(self, payload: dict[str, Any]) -> str:
        """Make a single request with retries."""
        url = f"{self.base_url}/completion"
        last_error = None

        for attempt in range(self.max_retries):
            # Check if server is still running
            if self._process and self._process.poll() is not None:
                stderr_output = self._get_recent_stderr()
                raise RuntimeError(
                    f"llama-server died during request. "
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
                content = data.get("content", "")
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

                if status == 503:
                    # Server busy, retry
                    last_error = "Server busy"
                elif status in (500, 502, 504):
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
        Generate text from multiple prompts using concurrent requests.

        The llama-server handles batching internally via cont_batching.
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
            logger.info("Stopping llama-server...")

            # Try graceful shutdown first
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

        # Wait for output threads to finish
        for thread in [self._stdout_thread, self._stderr_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=2)

        # Close HTTP session
        self._session.close()

        # Remove from active instances
        if self in LlamaCppServerBackend._active_instances:
            LlamaCppServerBackend._active_instances.remove(self)

        logger.debug("LlamaCppServerBackend closed")

    def __del__(self):
        """Ensure cleanup on garbage collection."""
        try:
            if hasattr(self, '_process') and self._process is not None:
                self.close()
        except Exception:
            pass
