# utils/db_schema.py

"""
Defines the database schema for the Creative Writing Benchmark using SQLAlchemy ORM.
This schema replaces the previous JSON file-based storage, providing better
scalability, concurrency, and data integrity. It includes tables for runs,
tasks, individual judge results, ELO data, and judge model configurations.
"""

import enum
from typing import Optional

from sqlalchemy import (
    create_engine, Integer, String, Float, DateTime, Boolean,
    ForeignKey, JSON, UniqueConstraint, Index, Text, Enum
)
from sqlalchemy.orm import relationship, declarative_base, Mapped, mapped_column
from sqlalchemy.sql import func

# Base class for all ORM models
Base = declarative_base()


class SubmissionStatus(str, enum.Enum):
    SUBMITTED = "SUBMITTED"
    QUEUED = "QUEUED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"


class Run(Base):
    """Represents a single benchmark run."""
    __tablename__ = 'runs'
    run_key: Mapped[str] = mapped_column(String, primary_key=True)
    test_model: Mapped[str] = mapped_column(String, nullable=False)
    start_time: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True), server_default=func.now())
    end_time: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False, default='initializing', index=True)
    run_config: Mapped[dict] = mapped_column(JSON, nullable=False)
    results: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    tasks: Mapped[list["Task"]] = relationship("Task", back_populates="run", cascade="all, delete-orphan")
    elo_comparisons: Mapped[list["EloComparison"]] = relationship("EloComparison", back_populates="run", cascade="all, delete-orphan")


class Task(Base):
    """Represents a single creative writing task for a specific prompt and iteration.

    For multi-turn writing tasks, model_responses stores a list of turn dictionaries:
    [
        {
            "turn_type": "planning" | "chapter",
            "turn_index": 0,  # 0 for planning, 1-N for chapters
            "user_prompt": "...",  # The user prompt for this turn
            "assistant_response": "...",  # The model's response (None if not yet generated)
            "status": "pending" | "generating" | "generated" | "error",
            "error": "..." | None,
            "chapter_number": N | None,  # Only for chapter turns
        },
        ...
    ]
    """
    __tablename__ = 'tasks'
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_key: Mapped[str] = mapped_column(String, ForeignKey('runs.run_key'), nullable=False)
    prompt_id: Mapped[str] = mapped_column(String, nullable=False)
    iteration_index: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default='initialized', index=True)

    # Legacy single-turn response field (kept for backward compatibility)
    model_response: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Multi-turn responses for longform writing tasks
    # List of dicts with turn_type, turn_index, user_prompt, assistant_response, status, error, chapter_number
    model_responses: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)

    error_message: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    # Stores aggregated scores after ensemble judging
    aggregated_scores: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    run: Mapped["Run"] = relationship("Run", back_populates="tasks")
    judge_results: Mapped[list["JudgeResult"]] = relationship("JudgeResult", back_populates="task", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint('run_key', 'prompt_id', 'iteration_index', name='_run_prompt_iter_uc'),
        Index('ix_tasks_run_key_status', 'run_key', 'status'),
    )


class JudgeResult(Base):
    """Stores the result from a single judge in an ensemble for a single task."""
    __tablename__ = 'judge_results'
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    task_id: Mapped[int] = mapped_column(Integer, ForeignKey('tasks.id'), nullable=False, index=True)
    judge_model_name: Mapped[str] = mapped_column(String, nullable=False)
    judge_order_index: Mapped[int] = mapped_column(Integer, nullable=False)
    raw_judge_text: Mapped[Optional[str]] = mapped_column(Text)
    judge_scores: Mapped[Optional[dict]] = mapped_column(JSON)

    task: Mapped["Task"] = relationship("Task", back_populates="judge_results")


class EloComparison(Base):
    """Stores a single pairwise ELO comparison, aggregated from a judge ensemble."""
    __tablename__ = 'elo_comparisons'
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_key: Mapped[Optional[str]] = mapped_column(String, ForeignKey('runs.run_key'), nullable=True)
    item_id: Mapped[str] = mapped_column(String, nullable=False)
    model_a: Mapped[str] = mapped_column(String, nullable=False)
    model_a_iteration_id: Mapped[str] = mapped_column(String, nullable=False)
    model_b: Mapped[str] = mapped_column(String, nullable=False)
    model_b_iteration_id: Mapped[str] = mapped_column(String, nullable=False)

    # Aggregated result from the judge ensemble
    aggregated_judge_responses: Mapped[Optional[dict]] = mapped_column(JSON)
    aggregated_plus_for_a: Mapped[Optional[int]] = mapped_column(Integer)
    aggregated_plus_for_b: Mapped[Optional[int]] = mapped_column(Integer)
    fraction_for_a: Mapped[Optional[float]] = mapped_column(Float)

    run: Mapped[Optional["Run"]] = relationship("Run", back_populates="elo_comparisons")

    __table_args__ = (
        Index('ix_elo_model_pair', 'model_a', 'model_b'),
    )


class EloRating(Base):
    """Stores the calculated ELO rating for a model."""
    __tablename__ = 'elo_ratings'
    model_name: Mapped[str] = mapped_column(String, primary_key=True)
    elo: Mapped[Optional[float]] = mapped_column(Float)
    elo_norm: Mapped[Optional[float]] = mapped_column(Float)
    sigma: Mapped[Optional[float]] = mapped_column(Float)
    ci_low: Mapped[Optional[float]] = mapped_column(Float)
    ci_high: Mapped[Optional[float]] = mapped_column(Float)
    ci_low_norm: Mapped[Optional[float]] = mapped_column(Float)
    ci_high_norm: Mapped[Optional[float]] = mapped_column(Float)
    last_updated: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())


class JudgeModel(Base):
    """Stores judge model configurations in the database, which can be overridden by a local YAML file."""
    __tablename__ = 'judge_models'
    name: Mapped[str] = mapped_column(String, primary_key=True)
    model_id: Mapped[str] = mapped_column(String, nullable=False)
    provider: Mapped[str] = mapped_column(String, nullable=False, default='openai')
    api_key: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    base_url: Mapped[str] = mapped_column(String, nullable=False)
    system_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class RunLog(Base):
    """Periodic stdout/stderr snapshots for a run, appended by the controller."""
    __tablename__ = 'run_logs'
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_key: Mapped[str] = mapped_column(String, ForeignKey('runs.run_key'), nullable=False, index=True)
    ts: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    stream: Mapped[str] = mapped_column(String, nullable=False)
    data: Mapped[str] = mapped_column(Text, nullable=False)

    __table_args__ = (
        Index('ix_run_logs_run_key_ts', 'run_key', 'ts'),
    )


# --- New tables for submission queue system ---

class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    email: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    auth_provider: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    auth_subject: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_ip: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    is_banned: Mapped[bool] = mapped_column(Boolean, default=False)
    role: Mapped[Optional[str]] = mapped_column(String, nullable=True)


class Submission(Base):
    __tablename__ = "submissions"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[Optional[str]] = mapped_column(ForeignKey("users.id"), index=True, nullable=True)
    created_ip: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    status: Mapped[SubmissionStatus] = mapped_column(Enum(SubmissionStatus), index=True, default=SubmissionStatus.SUBMITTED)
    params: Mapped[dict] = mapped_column(JSON, nullable=False)
    priority_score: Mapped[int] = mapped_column(Integer, default=0)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_runtime_sec: Mapped[int] = mapped_column(Integer, default=10800)
    run_key: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    runpod_pod_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    started_at: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True), nullable=True)
    error_msg: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

    user: Mapped[Optional["User"]] = relationship("User")


class Like(Base):
    __tablename__ = "likes"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    submission_id: Mapped[str] = mapped_column(ForeignKey("submissions.id"), index=True)
    user_id: Mapped[Optional[str]] = mapped_column(ForeignKey("users.id"), nullable=True)
    ip_hash: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("submission_id", "ip_hash", name="uq_like_submission_iphash"),
        UniqueConstraint("submission_id", "user_id", name="uq_like_submission_user"),
    )


class QueueToken(Base):
    __tablename__ = "queue_tokens"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[Optional[str]] = mapped_column(ForeignKey("users.id"), index=True)
    window_start: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True), index=True, server_default=func.now())
    submit_count: Mapped[int] = mapped_column(Integer, default=1)


class LeaderboardCache(Base):
    __tablename__ = "leaderboard_cache"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    snapshot_at: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True), server_default=func.now())
    data: Mapped[dict] = mapped_column(JSON, nullable=False)


class Setting(Base):
    __tablename__ = "settings"
    key: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[dict] = mapped_column(JSON, nullable=False)


class EventLog(Base):
    __tablename__ = "event_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_type: Mapped[str] = mapped_column(String, index=True)
    submission_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    user_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    ip: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    details: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


class Heartbeat(Base):
    __tablename__ = "heartbeat"
    # Singleton row: id always 1
    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    host: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    pid: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    updated_at: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    details: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)


# Composite index for submission queries
Index("ix_submissions_user_status", Submission.user_id, Submission.status)
