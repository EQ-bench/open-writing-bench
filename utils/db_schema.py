# utils/db_schema.py

"""
Defines the database schema for the Creative Writing Benchmark using SQLAlchemy ORM.
This schema replaces the previous JSON file-based storage, providing better
scalability, concurrency, and data integrity. It includes tables for runs,
tasks, individual judge results, ELO data, and judge model configurations.
"""

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime,
    ForeignKey, JSON, UniqueConstraint, Index, Text
)
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func

# Base class for all ORM models
Base = declarative_base()

class Run(Base):
    """Represents a single benchmark run."""
    __tablename__ = 'runs'
    run_key = Column(String, primary_key=True)
    test_model = Column(String, nullable=False)
    start_time = Column(DateTime(timezone=True), server_default=func.now())
    end_time = Column(DateTime(timezone=True), nullable=True)
    status = Column(String, nullable=False, default='initializing', index=True)
    run_config = Column(JSON, nullable=False)
    results = Column(JSON, nullable=True) # To store final benchmark_results, elo, etc.

    tasks = relationship("Task", back_populates="run", cascade="all, delete-orphan")
    elo_comparisons = relationship("EloComparison", back_populates="run", cascade="all, delete-orphan")

class Task(Base):
    """Represents a single creative writing task for a specific prompt and iteration."""
    __tablename__ = 'tasks'
    id = Column(Integer, primary_key=True)
    run_key = Column(String, ForeignKey('runs.run_key'), nullable=False)
    prompt_id = Column(String, nullable=False)
    iteration_index = Column(Integer, nullable=False)
    status = Column(String, nullable=False, default='initialized', index=True)
    model_response = Column(Text, nullable=True)
    error_message = Column(String, nullable=True)
    
    # Stores aggregated scores after ensemble judging
    aggregated_scores = Column(JSON, nullable=True)

    run = relationship("Run", back_populates="tasks")
    judge_results = relationship("JudgeResult", back_populates="task", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint('run_key', 'prompt_id', 'iteration_index', name='_run_prompt_iter_uc'),
        Index('ix_tasks_run_key_status', 'run_key', 'status'),
    )

class JudgeResult(Base):
    """Stores the result from a single judge in an ensemble for a single task."""
    __tablename__ = 'judge_results'
    id = Column(Integer, primary_key=True)
    task_id = Column(Integer, ForeignKey('tasks.id'), nullable=False, index=True)
    judge_model_name = Column(String, nullable=False) # The config name, e.g., 'gpt-4o-strict-judge'
    judge_order_index = Column(Integer, nullable=False) # To preserve ensemble order
    raw_judge_text = Column(Text)
    judge_scores = Column(JSON) # Parsed scores from this specific judge

    task = relationship("Task", back_populates="judge_results")

class EloComparison(Base):
    """Stores a single pairwise ELO comparison, aggregated from a judge ensemble."""
    __tablename__ = 'elo_comparisons'
    id = Column(Integer, primary_key=True)
    run_key = Column(String, ForeignKey('runs.run_key'), nullable=True) # Can be from multiple runs
    item_id = Column(String, nullable=False)
    model_a = Column(String, nullable=False)
    model_a_iteration_id = Column(String, nullable=False)
    model_b = Column(String, nullable=False)
    model_b_iteration_id = Column(String, nullable=False)
    
    # Aggregated result from the judge ensemble
    aggregated_judge_responses = Column(JSON) # Store raw responses from all judges
    aggregated_plus_for_a = Column(Integer)
    aggregated_plus_for_b = Column(Integer)
    fraction_for_a = Column(Float)

    run = relationship("Run", back_populates="elo_comparisons")

    __table_args__ = (
        Index('ix_elo_model_pair', 'model_a', 'model_b'),
    )

class EloRating(Base):
    """Stores the calculated ELO rating for a model."""
    __tablename__ = 'elo_ratings'
    model_name = Column(String, primary_key=True)
    elo = Column(Float)
    elo_norm = Column(Float)
    sigma = Column(Float)
    ci_low = Column(Float)
    ci_high = Column(Float)
    ci_low_norm = Column(Float)
    ci_high_norm = Column(Float)
    last_updated = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())

class JudgeModel(Base):
    """Stores judge model configurations in the database, which can be overridden by a local YAML file."""
    __tablename__ = 'judge_models'
    name = Column(String, primary_key=True) # e.g., 'gpt-4o-judge'
    model_id = Column(String, nullable=False) # e.g., 'gpt-4o'
    provider = Column(String, nullable=False, default='openai')
    api_key = Column(String, nullable=True) # Can be null if managed externally (e.g., via env var)
    base_url = Column(String, nullable=False)
    system_prompt = Column(Text, nullable=True)

class RunLog(Base):
    """Periodic stdout/stderr snapshots for a run, appended by the controller."""
    __tablename__ = 'run_logs'
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_key = Column(String, ForeignKey('runs.run_key'), nullable=False, index=True)
    ts = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    stream = Column(String, nullable=False)  # 'stdout' or 'stderr'
    data = Column(Text, nullable=False)

    __table_args__ = (
        Index('ix_run_logs_run_key_ts', 'run_key', 'ts'),
    )
