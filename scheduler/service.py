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

from .config import SchedulerConfig, load_config, get_queue_limits
from .queue_order import (
    SubmissionData, QueueLimits, QueueDecision,
    order_queue, extract_cost_from_results
)

# Global config reference, set by main() or Scheduler
_scheduler_config: Optional[SchedulerConfig] = None

# Load environment before importing db
load_dotenv()

from utils.db_connector import db
from utils.db_schema import Submission, SubmissionStatus, RunLog, EventLog, Run, EloRating, User

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

        # Build backend config from vllmParams - pass through all specified params
        vllm_params = params.get("vllmParams", {})
        backend_config = dict(vllm_params) if vllm_params else {}

        cmd = [
            sys.executable, "-m", "open_writing_bench",
            "--test-model", model_id,
            "--test-provider", test_provider,
            "--judge-models", judges_str,
            "--threads", str(self.config.default_threads),
            "--verbosity", self.config.default_verbosity,
            "--run-id", submission.id,  # Use submission ID as run ID
            "--ensemble-mode", "split",  # Distribute tasks across judges
            "--disable-elo-reasoning",  # Reduce token usage in ELO comparisons
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


def kill_sandbox_vllm_processes(sandbox_user: str) -> bool:
    """Kill any vLLM processes running under the sandbox user.

    Returns:
        True if any processes were killed, False otherwise.
    """
    killed = False
    try:
        # First, try to kill by user (most reliable for sandboxed processes)
        result = subprocess.run(
            ["pkill", "-u", sandbox_user],
            capture_output=True,
            timeout=10
        )
        if result.returncode == 0:
            killed = True
            logger.info(f"Killed processes for user {sandbox_user}")

        # Also try pkill -f vllm for any processes that might have escaped
        result = subprocess.run(
            ["pkill", "-f", "vllm"],
            capture_output=True,
            timeout=10
        )
        if result.returncode == 0:
            killed = True
            logger.info("Killed vllm processes (by command pattern)")

    except subprocess.TimeoutExpired:
        logger.warning("Timeout while killing processes, forcing with SIGKILL...")
        try:
            subprocess.run(["pkill", "-9", "-u", sandbox_user], capture_output=True, timeout=5)
            subprocess.run(["pkill", "-9", "-f", "vllm"], capture_output=True, timeout=5)
            killed = True
        except Exception as e:
            logger.error(f"Failed to force-kill processes: {e}")
    except Exception as e:
        logger.warning(f"Failed to kill vLLM processes: {e}")

    return killed


def cleanup_after_job(config: SchedulerConfig):
    """Clean up after a job completes."""
    if config.kill_vllm_processes:
        logger.info(f"Killing any remaining vLLM processes (sandbox user: {config.sandbox_user})...")
        kill_sandbox_vllm_processes(config.sandbox_user)

    if config.clear_hf_cache:
        logger.info("Clearing old HuggingFace cache directories...")
        hf_cache_dirs = [
            Path("/workspace/mounted/sam/hf/hub"),
            Path("/workspace/mounted/sam/hf/xet"),
        ]
        cutoff_time = time.time() - (4 * 60 * 60)  # 4 hours ago
        for hf_cache_root in hf_cache_dirs:
            if hf_cache_root.exists():
                for subdir in hf_cache_root.iterdir():
                    if subdir.is_dir():
                        try:
                            dir_mtime = subdir.stat().st_mtime
                            if dir_mtime < cutoff_time:
                                shutil.rmtree(subdir)
                                logger.info(f"Cleared old cache dir: {subdir}")
                        except Exception as e:
                            logger.warning(f"Failed to clear {subdir}: {e}")


def load_submissions_for_queue_ordering(
    window_hours: int = 24
) -> tuple[list[SubmissionData], list[SubmissionData]]:
    """
    Load submission data from DB for queue ordering.

    Returns:
        (pending_submissions, all_submissions_in_window)
    """
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(hours=window_hours)

    with db.get_session() as session:
        # Load all submissions in the window (for stats calculation)
        all_subs = session.execute(
            select(Submission)
            .where(Submission.created_at >= window_start)
        ).scalars().all()

        # Also load any runs to get results/costs
        run_results: dict[str, dict] = {}
        if all_subs:
            run_keys = [s.id for s in all_subs if s.run_key]
            if run_keys:
                runs = session.execute(
                    select(Run).where(Run.run_key.in_(run_keys))
                ).scalars().all()
                for run in runs:
                    run_results[run.run_key] = run.results

        # Load user roles for admin bypass
        user_roles: dict[str, str] = {}
        user_ids = {s.user_id for s in all_subs if s.user_id}
        if user_ids:
            users = session.execute(
                select(User).where(User.id.in_(user_ids))
            ).scalars().all()
            for user in users:
                if user.role:
                    user_roles[user.id] = user.role

        # Convert to SubmissionData
        pending: list[SubmissionData] = []
        all_data: list[SubmissionData] = []

        for sub in all_subs:
            data = SubmissionData(
                id=sub.id,
                user_id=sub.user_id,
                created_ip=sub.created_ip,
                created_at=sub.created_at,
                status=sub.status.value if isinstance(sub.status, SubmissionStatus) else str(sub.status),
                started_at=sub.started_at,
                finished_at=sub.finished_at,
                results=run_results.get(sub.id),
                user_role=user_roles.get(sub.user_id)
            )
            all_data.append(data)

            # Pending = SUBMITTED or QUEUED
            if sub.status in (SubmissionStatus.SUBMITTED, SubmissionStatus.QUEUED):
                pending.append(data)

        return pending, all_data


def get_next_submission_with_ordering(limits: Optional[QueueLimits] = None) -> tuple[Optional[Submission], Optional[QueueDecision]]:
    """
    Get the next submission to process using queue ordering.

    Returns:
        (submission, decision) - submission is None if queue is empty or all are held
    """
    if limits is None:
        if _scheduler_config:
            limits = get_queue_limits(_scheduler_config)
        else:
            limits = QueueLimits()

    now = datetime.now(timezone.utc)
    pending, all_data = load_submissions_for_queue_ordering(limits.window_hours)

    if not pending:
        return None, None

    # Get ordered decisions
    decisions = order_queue(pending, all_data, limits=limits, now=now)

    # Find first processable submission
    for decision in decisions:
        if decision.action == "process":
            # Fetch the actual Submission object
            with db.get_session() as session:
                submission = session.get(Submission, decision.submission_id)
                if submission:
                    session.expunge(submission)
                    return submission, decision
        elif decision.action == "hold":
            # Log hold reason
            logger.debug(f"Submission {decision.submission_id} held: {decision.hold_reason}")

    return None, None


def update_priority_scores(limits: Optional[QueueLimits] = None) -> int:
    """
    Update priority_score for all pending submissions.

    Returns:
        Number of submissions updated.
    """
    if limits is None:
        if _scheduler_config:
            limits = get_queue_limits(_scheduler_config)
        else:
            limits = QueueLimits()

    now = datetime.now(timezone.utc)
    pending, all_data = load_submissions_for_queue_ordering(limits.window_hours)

    if not pending:
        return 0

    # Get ordered decisions
    decisions = order_queue(pending, all_data, limits=limits, now=now)

    # Build a map of submission_id -> new score
    new_scores: dict[str, float] = {}
    for decision in decisions:
        # For held submissions, use a negative score to push them to the back
        if decision.action == "hold":
            new_scores[decision.submission_id] = -1.0
        else:
            new_scores[decision.submission_id] = decision.priority_score

    # Update in DB, only if changed
    updated = 0
    with db.get_session() as session:
        for sub_id, new_score in new_scores.items():
            submission = session.get(Submission, sub_id)
            if submission and submission.priority_score != int(new_score):
                submission.priority_score = int(new_score)
                updated += 1

    return updated


def get_next_submission() -> Optional[Submission]:
    """Get the next submission to process (SUBMITTED or QUEUED status)."""
    # Use the new queue ordering
    submission, decision = get_next_submission_with_ordering()

    if decision and decision.action == "process":
        logger.debug(f"Selected submission {decision.submission_id} with score {decision.priority_score:.2f}")

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
        self._priority_updater_thread: Optional[threading.Thread] = None
        self._last_priority_update = 0.0

        # Set global config for queue ordering functions
        global _scheduler_config
        _scheduler_config = config

    def _handle_signal(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self._shutdown.set()
        self.runner.kill()

    def _priority_updater_loop(self):
        """Background thread that periodically updates priority scores."""
        logger.info("Priority score updater started")
        interval = self.config.priority_update_interval_sec

        while not self._shutdown.is_set():
            try:
                limits = get_queue_limits(self.config)
                updated = update_priority_scores(limits)
                if updated > 0:
                    logger.info(f"Updated priority scores for {updated} submissions")
            except Exception as e:
                logger.error(f"Error updating priority scores: {e}")

            # Wait for next update interval or shutdown
            self._shutdown.wait(timeout=interval)

        logger.info("Priority score updater stopped")

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

        # Pre-run cleanup: ensure no stale vLLM processes from previous runs
        if self.config.kill_vllm_processes:
            print("Pre-run cleanup: checking for stale vLLM processes...")
            if kill_sandbox_vllm_processes(self.config.sandbox_user):
                print("Killed stale processes, waiting for cleanup...")
                time.sleep(2)  # Give processes time to fully terminate

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
        logger.info(f"Priority update interval: {self.config.priority_update_interval_sec}s")
        logger.info(f"Verbose mode: {self.config.verbose}")

        # Log queue limits
        limits = get_queue_limits(self.config)
        logger.info(f"Queue limits: ${limits.max_cost_per_user_24h}/user, ${limits.max_cost_per_ip_24h}/IP, "
                   f"{limits.max_runtime_per_user_24h_sec/3600:.0f}h runtime/user")

        print(f"Polling for jobs every {self.config.poll_interval_sec}s...")
        print(f"Priority scores updated every {self.config.priority_update_interval_sec}s")
        print()

        # Start background priority updater thread
        self._priority_updater_thread = threading.Thread(
            target=self._priority_updater_loop,
            daemon=True,
            name="priority-updater"
        )
        self._priority_updater_thread.start()

        # Do an initial priority score update
        try:
            updated = update_priority_scores()
            logger.info(f"Initial priority score update: {updated} submissions")
        except Exception as e:
            logger.error(f"Error in initial priority update: {e}")

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

        # Wait for priority updater to stop
        if self._priority_updater_thread and self._priority_updater_thread.is_alive():
            self._priority_updater_thread.join(timeout=5)

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
