# core/progress.py

"""
Thread-safe progress tracking for benchmark runs.

Workers update in-memory counters atomically, and the main thread
periodically flushes snapshots to the database to avoid expensive queries.
"""

import threading
from dataclasses import dataclass, field
from typing import Optional

from utils.db_connector import db


@dataclass
class RunProgress:
    """Thread-safe progress tracker for a benchmark run."""

    run_key: str
    total_tasks: int = 0
    total_turns: int = 0  # For multiturn: total_tasks * (1 + num_chapters)

    # Internal counters protected by lock
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _completed_turns: int = 0
    _completed_tasks: int = 0
    _generation_errors: int = 0

    # Rubric judging counters
    _rubric_total: int = 0
    _rubric_completed: int = 0
    _rubric_errors: int = 0

    # ELO judging counters
    _elo_current_stage: int = 0
    _elo_total_stages: int = 0
    _elo_comparisons_completed: int = 0

    # Track phase for context
    _phase: str = "generation"  # "generation", "rubric_judging", or "elo_judging"

    def set_phase(self, phase: str):
        """Set current phase ('generation', 'rubric_judging', or 'elo_judging')."""
        with self._lock:
            self._phase = phase

    # --- Generation phase increments ---

    def inc_completed_turns(self, n: int = 1):
        """Increment completed turns (call after each turn completes)."""
        with self._lock:
            self._completed_turns += n

    def inc_completed_tasks(self, n: int = 1):
        """Increment completed generation tasks."""
        with self._lock:
            self._completed_tasks += n

    def inc_generation_errors(self, n: int = 1):
        """Increment generation error count."""
        with self._lock:
            self._generation_errors += n

    # --- Rubric judging phase ---

    def set_rubric_total(self, total: int):
        """Set total tasks to judge with rubric."""
        with self._lock:
            self._rubric_total = total

    def inc_rubric_completed(self, n: int = 1):
        """Increment rubric judged tasks count."""
        with self._lock:
            self._rubric_completed += n

    def inc_rubric_errors(self, n: int = 1):
        """Increment rubric judging error count."""
        with self._lock:
            self._rubric_errors += n

    # --- ELO judging phase ---

    def set_elo_stages(self, total_stages: int):
        """Set total number of ELO stages."""
        with self._lock:
            self._elo_total_stages = total_stages
            self._elo_current_stage = 0

    def set_elo_stage(self, stage: int):
        """Set current ELO stage (1-indexed)."""
        with self._lock:
            self._elo_current_stage = stage

    def inc_elo_comparisons(self, n: int = 1):
        """Increment ELO comparisons completed."""
        with self._lock:
            self._elo_comparisons_completed += n

    # --- Snapshots ---

    def generation_snapshot(self) -> dict:
        """Get current generation progress as dict."""
        with self._lock:
            return {
                "total_tasks": self.total_tasks,
                "total_turns": self.total_turns,
                "completed_turns": self._completed_turns,
                "completed_tasks": self._completed_tasks,
                "error_tasks": self._generation_errors,
            }

    def judging_snapshot(self) -> dict:
        """Get current judging progress as dict."""
        with self._lock:
            return {
                "rubric": {
                    "total_tasks": self._rubric_total,
                    "completed_tasks": self._rubric_completed,
                    "error_tasks": self._rubric_errors,
                },
                "elo": {
                    "current_stage": self._elo_current_stage,
                    "total_stages": self._elo_total_stages,
                    "comparisons_completed": self._elo_comparisons_completed,
                },
            }

    def flush_to_db(self):
        """Write current progress to database."""
        gen_progress = self.generation_snapshot()
        judge_progress = self.judging_snapshot()
        db.update_run(self.run_key, {
            "generation_progress": gen_progress,
            "judging_progress": judge_progress,
        })

    def flush_generation_to_db(self):
        """Write only generation progress to database."""
        db.update_run(self.run_key, {
            "generation_progress": self.generation_snapshot(),
        })

    def flush_judging_to_db(self):
        """Write only judging progress to database."""
        db.update_run(self.run_key, {
            "judging_progress": self.judging_snapshot(),
        })
