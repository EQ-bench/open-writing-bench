# scheduler/service.py
"""
Scheduler service for managing open-writing-bench jobs.

This service:
1. Polls the submissions table for QUEUED jobs
2. Runs one job at a time using the open_writing_bench CLI
3. Streams stdout/stderr to run_logs table periodically
4. Handles timeouts (2hr default hard limit)
5. Cleans up vLLM processes and HF cache after each job
6. Uses a file lock to ensure only one scheduler instance runs
"""

import fcntl
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from sqlalchemy import select, func

from .config import SchedulerConfig, load_config

# Load environment before importing db
load_dotenv()

from utils.db_connector import db
from utils.db_schema import Submission, SubmissionStatus, RunLog, EventLog, Run, EloRating

logger = logging.getLogger(__name__)


class SchedulerLock:
    """File-based lock to ensure only one scheduler instance runs."""

    def __init__(self, lock_path: str | Path = "/tmp/owb-scheduler.lock"):
        self.lock_path = Path(lock_path)
        self._lock_file = None

    def acquire(self) -> bool:
        """Try to acquire the lock. Returns True if successful."""
        try:
            self._lock_file = open(self.lock_path, "w")
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Write PID for debugging
            self._lock_file.write(str(os.getpid()))
            self._lock_file.flush()
            return True
        except (IOError, OSError):
            if self._lock_file:
                self._lock_file.close()
                self._lock_file = None
            return False

    def release(self):
        """Release the lock."""
        if self._lock_file:
            try:
                fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
                self._lock_file.close()
            except Exception:
                pass
            self._lock_file = None

    def __enter__(self):
        if not self.acquire():
            raise RuntimeError("Another scheduler instance is already running")
        return self

    def __exit__(self, *args):
        self.release()


class JobRunner:
    """Runs a single open-writing-bench job and streams logs."""

    def __init__(self, config: SchedulerConfig):
        self.config = config
        self._process: Optional[subprocess.Popen] = None
        self._stdout_buffer: list[str] = []
        self._stderr_buffer: list[str] = []
        self._lock = threading.Lock()
        self._stop_flag = threading.Event()
        self._last_printed_stdout = 0
        self._last_printed_stderr = 0

    def _build_command(self, submission: Submission) -> list[str]:
        """Build the CLI command from submission params."""
        params = submission.params or {}

        # Extract model info
        model_id = params.get("modelId", "")
        if not model_id:
            # Try modelInfo.modelId as fallback
            model_info = params.get("modelInfo", {})
            model_id = model_info.get("modelId", model_info.get("id", ""))

        if not model_id:
            raise ValueError("No model ID found in submission params")

        # Determine test provider
        model_type = params.get("modelType", "huggingface").lower()
        if model_type in ("huggingface", "hf", "vllm"):
            test_provider = "vllm"
        else:
            test_provider = self.config.default_test_provider

        # Get judge models
        judges = params.get("judges", self.config.default_judges)
        if isinstance(judges, list):
            judges_str = ",".join(judges)
        else:
            judges_str = str(judges)

        # Build backend config from vllmParams
        vllm_params = params.get("vllmParams", {})
        backend_config = {}
        if vllm_params:
            # Copy relevant params
            for key in ["gpu_memory_utilization", "max_model_len", "dtype", "quantization",
                        "enforce_eager", "tensor_parallel_size", "ENV_VARS"]:
                if key in vllm_params:
                    backend_config[key] = vllm_params[key]

        cmd = [
            sys.executable, "-m", "open_writing_bench",
            "--test-model", model_id,
            "--test-provider", test_provider,
            "--judge-models", judges_str,
            "--threads", str(self.config.default_threads),
            "--verbosity", self.config.default_verbosity,
            "--run-id", submission.id,  # Use submission ID as run ID
        ]

        if backend_config:
            cmd.extend(["--backend-config", json.dumps(backend_config)])

        return cmd

    def _stream_reader(self, stream, buffer: list[str], name: str):
        """Read from a stream and append to buffer."""
        try:
            for line in iter(stream.readline, ""):
                if self._stop_flag.is_set():
                    break
                with self._lock:
                    buffer.append(line)
                # In verbose mode, print job output to console immediately
                if self.config.verbose:
                    prefix = "[JOB]" if name == "stdout" else "[JOB ERR]"
                    print(f"{prefix} {line}", end="", flush=True)
        except Exception as e:
            logger.error(f"Error reading {name}: {e}")
        finally:
            stream.close()

    def _write_logs_to_db(self, run_key: str):
        """Write accumulated logs to the run_logs table (upsert by run_key + stream)."""
        with self._lock:
            stdout_data = "".join(self._stdout_buffer)
            stderr_data = "".join(self._stderr_buffer)

        if not stdout_data and not stderr_data:
            return

        try:
            with db.get_session() as session:
                now = datetime.now(timezone.utc)

                # Upsert stdout log
                if stdout_data:
                    existing_stdout = session.execute(
                        select(RunLog).where(
                            RunLog.run_key == run_key,
                            RunLog.stream == "stdout"
                        )
                    ).scalar_one_or_none()

                    if existing_stdout:
                        existing_stdout.data = stdout_data
                        existing_stdout.ts = now
                    else:
                        session.add(RunLog(
                            run_key=run_key,
                            ts=now,
                            stream="stdout",
                            data=stdout_data
                        ))

                # Upsert stderr log
                if stderr_data:
                    existing_stderr = session.execute(
                        select(RunLog).where(
                            RunLog.run_key == run_key,
                            RunLog.stream == "stderr"
                        )
                    ).scalar_one_or_none()

                    if existing_stderr:
                        existing_stderr.data = stderr_data
                        existing_stderr.ts = now
                    else:
                        session.add(RunLog(
                            run_key=run_key,
                            ts=now,
                            stream="stderr",
                            data=stderr_data
                        ))
        except Exception as e:
            logger.error(f"Failed to write logs to DB: {e}")

    def run(self, submission: Submission) -> tuple[bool, str]:
        """
        Run the benchmark for a submission.

        Returns:
            (success, error_message) tuple
        """
        run_key = submission.id
        self._stdout_buffer = []
        self._stderr_buffer = []
        self._stop_flag.clear()

        try:
            cmd = self._build_command(submission)
        except ValueError as e:
            return False, str(e)

        logger.info(f"Starting job {run_key}")
        logger.debug(f"Command: {' '.join(cmd)}")
        print(f"Command: {' '.join(cmd[:6])}...")  # Show truncated command
        if self.config.verbose:
            print(f"Full command: {' '.join(cmd)}")

        # Start the subprocess
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                cwd=Path(__file__).parent.parent,  # Run from project root
            )
            logger.info(f"Subprocess started with PID {self._process.pid}")
            print(f"Process started (PID: {self._process.pid})")
        except Exception as e:
            logger.error(f"Failed to start process: {e}")
            return False, f"Failed to start process: {e}"

        # Start reader threads
        stdout_thread = threading.Thread(
            target=self._stream_reader,
            args=(self._process.stdout, self._stdout_buffer, "stdout"),
            daemon=True
        )
        stderr_thread = threading.Thread(
            target=self._stream_reader,
            args=(self._process.stderr, self._stderr_buffer, "stderr"),
            daemon=True
        )
        stdout_thread.start()
        stderr_thread.start()

        # Monitor loop with periodic log streaming
        start_time = time.time()
        last_log_time = start_time

        while self._process.poll() is None:
            time.sleep(1)

            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > self.config.hard_timeout_sec:
                logger.warning(f"Job {run_key} exceeded timeout ({self.config.hard_timeout_sec}s)")
                self._stop_flag.set()
                self._process.terminate()
                try:
                    self._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                self._write_logs_to_db(run_key)
                return False, f"Timeout after {self.config.hard_timeout_sec} seconds"

            # Periodic log streaming
            if time.time() - last_log_time >= self.config.log_stream_interval_sec:
                self._write_logs_to_db(run_key)
                last_log_time = time.time()

        # Wait for reader threads to finish
        self._stop_flag.set()
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)

        # Final log flush
        self._write_logs_to_db(run_key)

        # Check exit code
        exit_code = self._process.returncode
        if exit_code == 0:
            return True, ""
        else:
            # Get last few lines of stderr for error message
            with self._lock:
                last_stderr = "".join(self._stderr_buffer[-20:])
            return False, f"Exit code {exit_code}: {last_stderr[:500]}"

    def kill(self):
        """Kill the running process if any."""
        if self._process and self._process.poll() is None:
            self._stop_flag.set()
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()


def cleanup_after_job(config: SchedulerConfig):
    """Clean up after a job completes."""
    if config.kill_vllm_processes:
        logger.info("Killing any remaining vLLM processes...")
        try:
            # Kill any python processes with vllm in the command line
            subprocess.run(
                ["pkill", "-f", "vllm"],
                capture_output=True,
                timeout=10
            )
        except Exception as e:
            logger.warning(f"Failed to kill vLLM processes: {e}")

    if config.clear_hf_cache:
        logger.info("Clearing HuggingFace cache...")
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        if cache_dir.exists():
            try:
                shutil.rmtree(cache_dir)
                logger.info("HuggingFace cache cleared")
            except Exception as e:
                logger.warning(f"Failed to clear HF cache: {e}")


def get_next_submission() -> Optional[Submission]:
    """Get the next submission to process (SUBMITTED or QUEUED status)."""
    with db.get_session() as session:
        submission = session.execute(
            select(Submission)
            .where(Submission.status.in_([SubmissionStatus.SUBMITTED, SubmissionStatus.QUEUED]))
            .order_by(Submission.priority_score.desc(), Submission.created_at.asc())
            .limit(1)
        ).scalar_one_or_none()

        if submission:
            # Detach from session for use outside
            session.expunge(submission)

        return submission


def check_model_already_rated(model_id: str) -> bool:
    """Check if a model already has an ELO rating in the database."""
    with db.get_session() as session:
        existing = session.get(EloRating, model_id)
        return existing is not None


def mark_submission_starting(submission_id: str) -> str:
    """Mark a submission as STARTING and return the run_key."""
    with db.get_session() as session:
        submission = session.get(Submission, submission_id)
        if not submission:
            raise ValueError(f"Submission {submission_id} not found")

        submission.status = SubmissionStatus.STARTING
        submission.started_at = datetime.now(timezone.utc)
        submission.run_key = submission_id  # Use submission ID as run key

        # Create the Run record
        run_config = {
            "submission_id": submission_id,
            "params": submission.params,
        }
        model_id = submission.params.get("modelId", "unknown")

        # Check if run already exists
        existing_run = session.get(Run, submission_id)
        if not existing_run:
            run = Run(
                run_key=submission_id,
                test_model=model_id,
                status="starting",
                run_config=run_config
            )
            session.add(run)

        session.add(EventLog(
            event_type="SUBMISSION_STARTING",
            submission_id=submission_id,
            details={"run_key": submission_id}
        ))

        return submission_id


def mark_submission_running(submission_id: str):
    """Mark a submission as RUNNING."""
    with db.get_session() as session:
        submission = session.get(Submission, submission_id)
        if submission:
            submission.status = SubmissionStatus.RUNNING
            session.add(EventLog(
                event_type="SUBMISSION_RUNNING",
                submission_id=submission_id,
            ))

        run = session.get(Run, submission_id)
        if run:
            run.status = "running"


def mark_submission_succeeded(submission_id: str):
    """Mark a submission as SUCCEEDED."""
    with db.get_session() as session:
        submission = session.get(Submission, submission_id)
        if submission:
            submission.status = SubmissionStatus.SUCCEEDED
            submission.finished_at = datetime.now(timezone.utc)
            session.add(EventLog(
                event_type="SUBMISSION_SUCCEEDED",
                submission_id=submission_id,
            ))

        run = session.get(Run, submission_id)
        if run:
            run.status = "completed"
            run.end_time = datetime.now(timezone.utc)


def mark_submission_failed(submission_id: str, error_msg: str, config: SchedulerConfig):
    """Mark a submission as FAILED or re-queue for retry."""
    with db.get_session() as session:
        submission = session.get(Submission, submission_id)
        if not submission:
            return

        submission.attempts += 1

        if submission.attempts < config.max_attempts:
            # Re-queue for retry
            submission.status = SubmissionStatus.QUEUED
            session.add(EventLog(
                event_type="SUBMISSION_RETRY",
                submission_id=submission_id,
                details={"attempts": submission.attempts, "error": error_msg[:500]}
            ))
        else:
            # Max attempts reached
            submission.status = SubmissionStatus.FAILED
            submission.finished_at = datetime.now(timezone.utc)
            submission.error_msg = error_msg
            session.add(EventLog(
                event_type="SUBMISSION_FAILED",
                submission_id=submission_id,
                details={"attempts": submission.attempts, "error": error_msg[:500]}
            ))

        run = session.get(Run, submission_id)
        if run:
            run.status = "failed" if submission.attempts >= config.max_attempts else "retry_queued"
            run.end_time = datetime.now(timezone.utc)


def mark_submission_timeout(submission_id: str):
    """Mark a submission as TIMEOUT."""
    with db.get_session() as session:
        submission = session.get(Submission, submission_id)
        if submission:
            submission.status = SubmissionStatus.TIMEOUT
            submission.finished_at = datetime.now(timezone.utc)
            submission.error_msg = "Job exceeded hard timeout limit"
            session.add(EventLog(
                event_type="SUBMISSION_TIMEOUT",
                submission_id=submission_id,
            ))

        run = session.get(Run, submission_id)
        if run:
            run.status = "timeout"
            run.end_time = datetime.now(timezone.utc)


class Scheduler:
    """Main scheduler service class."""

    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.runner = JobRunner(config)
        self._shutdown = threading.Event()

    def _handle_signal(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self._shutdown.set()
        self.runner.kill()

    def run_once(self) -> bool:
        """
        Check for and process one job.

        Returns:
            True if a job was processed, False if queue was empty.
        """
        logger.debug("Checking for pending submissions...")
        submission = get_next_submission()
        if not submission:
            logger.debug("No pending submissions found")
            return False

        submission_id = submission.id
        model_id = submission.params.get("modelId", "unknown") if submission.params else "unknown"
        print()
        print("-" * 60)
        print(f"Found submission: {submission_id}")
        print(f"Model: {model_id}")
        print("-" * 60)
        logger.info(f"Processing submission {submission_id} (model: {model_id})")

        # Check if model already has an ELO rating
        if check_model_already_rated(model_id):
            error_msg = f"Model '{model_id}' already has an ELO rating in the database. Duplicate submissions are not allowed."
            print(f"[SKIPPED] {error_msg}")
            logger.warning(error_msg)
            mark_submission_failed(submission_id, error_msg, self.config)
            return True

        try:
            run_key = mark_submission_starting(submission_id)
            logger.info(f"Submission {submission_id} marked as STARTING")
            mark_submission_running(submission_id)
            logger.info(f"Submission {submission_id} marked as RUNNING")
            print(f"Job started at {datetime.now(timezone.utc).isoformat()}")
            print()

            success, error_msg = self.runner.run(submission)

            print()
            if success:
                mark_submission_succeeded(submission_id)
                print(f"[SUCCESS] Submission {submission_id} completed successfully")
                logger.info(f"Submission {submission_id} completed successfully")
            elif "Timeout" in error_msg:
                mark_submission_timeout(submission_id)
                print(f"[TIMEOUT] Submission {submission_id} timed out")
                logger.warning(f"Submission {submission_id} timed out")
            else:
                mark_submission_failed(submission_id, error_msg, self.config)
                print(f"[FAILED] Submission {submission_id} failed: {error_msg[:200]}")
                logger.error(f"Submission {submission_id} failed: {error_msg}")

        except Exception as e:
            logger.exception(f"Unexpected error processing submission {submission_id}")
            print(f"[ERROR] Unexpected error: {e}")
            mark_submission_failed(submission_id, str(e), self.config)

        finally:
            print()
            print("Cleaning up...")
            cleanup_after_job(self.config)
            print("Cleanup complete.")
            print("-" * 60)

        return True

    def run(self):
        """Run the scheduler main loop."""
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        print("=" * 60)
        print("Open Writing Bench Scheduler")
        print("=" * 60)
        logger.info("Scheduler started")
        logger.info(f"Poll interval: {self.config.poll_interval_sec}s")
        logger.info(f"Hard timeout: {self.config.hard_timeout_sec}s")
        logger.info(f"Verbose mode: {self.config.verbose}")
        print(f"Polling for jobs every {self.config.poll_interval_sec}s...")
        print()

        while not self._shutdown.is_set():
            try:
                job_processed = self.run_once()

                if not job_processed:
                    # No job in queue, sleep before next poll
                    logger.debug(f"No jobs in queue, sleeping {self.config.poll_interval_sec}s...")
                    self._shutdown.wait(timeout=self.config.poll_interval_sec)
                # If a job was processed, immediately check for next job

            except Exception as e:
                logger.exception(f"Error in scheduler loop: {e}")
                self._shutdown.wait(timeout=self.config.poll_interval_sec)

        logger.info("Scheduler stopped")
        print("\nScheduler stopped.")


def main():
    """Main entry point for the scheduler service."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Scheduler service for open-writing-bench jobs. "
                    "Polls the submissions table and runs benchmark jobs one at a time."
    )
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="Path to scheduler.cfg configuration file"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process one job and exit (don't run as daemon)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True  # Override any existing config
    )
    # Also set level on root logger explicitly
    logging.getLogger().setLevel(log_level)
    # And on our module logger
    logger.setLevel(log_level)

    config = load_config(args.config)
    config.verbose = args.verbose  # Pass verbose flag to config

    # Ensure only one scheduler runs
    lock = SchedulerLock()
    if not lock.acquire():
        logger.error("Another scheduler instance is already running")
        sys.exit(1)

    try:
        scheduler = Scheduler(config)
        if args.once:
            scheduler.run_once()
        else:
            scheduler.run()
    finally:
        lock.release()


if __name__ == "__main__":
    main()
