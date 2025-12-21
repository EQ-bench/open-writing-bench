# scheduler/config.py
"""Configuration loader for the scheduler service."""

import configparser
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SchedulerConfig:
    """Scheduler configuration loaded from scheduler.cfg."""

    # Scheduler settings
    poll_interval_sec: int = 10
    log_stream_interval_sec: int = 10
    hard_timeout_sec: int = 7200  # 2 hours default
    max_attempts: int = 1
    priority_update_interval_sec: int = 60  # How often to update priority scores

    # Default job settings
    default_threads: int = 34
    default_verbosity: str = "DEBUG"
    default_test_provider: str = "vllm"
    default_judges: list[str] = None

    # Cleanup settings
    clear_hf_cache: bool = True
    kill_vllm_processes: bool = True
    sandbox_user: str = "vllm-sandbox"

    # Queue ordering settings (rate limiting and fairness)
    queue_max_cost_per_user_24h: float = 10.0      # $ limit per user per 24h
    queue_max_cost_per_ip_24h: float = 10.0        # $ limit per IP per 24h
    queue_max_runtime_per_user_24h_hours: float = 8.0  # Hours limit per user
    queue_max_runtime_per_ip_24h_hours: float = 8.0    # Hours limit per IP
    queue_max_concurrent_per_user: int = 3         # Max queued/running jobs per user
    queue_max_concurrent_per_ip: int = 5           # Max queued/running jobs per IP
    queue_window_hours: int = 24                   # Rolling window for limits
    queue_weight_low_cost: float = 10.0            # Priority weight for low cost usage
    queue_weight_low_runtime: float = 5.0          # Priority weight for low runtime usage

    # Runtime flags (set by CLI, not config file)
    verbose: bool = False

    def __post_init__(self):
        if self.default_judges is None:
            self.default_judges = ["grok-4.1-fast", "claude-haiku-4.5", "kimi-k2-0905"]


def load_config(config_path: str | Path | None = None) -> SchedulerConfig:
    """Load scheduler configuration from file.

    Args:
        config_path: Path to scheduler.cfg. If None, looks in current directory
                     and project root.

    Returns:
        SchedulerConfig with values from file or defaults.
    """
    if config_path is None:
        # Look for scheduler.cfg in common locations
        candidates = [
            Path("scheduler.cfg"),
            Path(__file__).parent.parent / "scheduler.cfg",
        ]
        for candidate in candidates:
            if candidate.exists():
                config_path = candidate
                break

    config = SchedulerConfig()

    if config_path and Path(config_path).exists():
        parser = configparser.ConfigParser()
        parser.read(config_path)

        # Scheduler section
        if parser.has_section("scheduler"):
            sched = parser["scheduler"]
            config.poll_interval_sec = sched.getint("poll_interval_sec", config.poll_interval_sec)
            config.log_stream_interval_sec = sched.getint("log_stream_interval_sec", config.log_stream_interval_sec)
            config.hard_timeout_sec = sched.getint("hard_timeout_sec", config.hard_timeout_sec)
            config.max_attempts = sched.getint("max_attempts", config.max_attempts)

        # Defaults section
        if parser.has_section("defaults"):
            defaults = parser["defaults"]
            config.default_threads = defaults.getint("threads", config.default_threads)
            config.default_verbosity = defaults.get("verbosity", config.default_verbosity)
            config.default_test_provider = defaults.get("test_provider", config.default_test_provider)
            judges_str = defaults.get("default_judges", "")
            if judges_str:
                config.default_judges = [j.strip() for j in judges_str.split(",") if j.strip()]

        # Cleanup section
        if parser.has_section("cleanup"):
            cleanup = parser["cleanup"]
            config.clear_hf_cache = cleanup.getboolean("clear_hf_cache", config.clear_hf_cache)
            config.kill_vllm_processes = cleanup.getboolean("kill_vllm_processes", config.kill_vllm_processes)
            config.sandbox_user = cleanup.get("sandbox_user", config.sandbox_user)

        # Queue section (rate limiting and fairness)
        if parser.has_section("queue"):
            queue = parser["queue"]
            config.priority_update_interval_sec = queue.getint("priority_update_interval_sec", config.priority_update_interval_sec)
            config.queue_max_cost_per_user_24h = queue.getfloat("max_cost_per_user_24h", config.queue_max_cost_per_user_24h)
            config.queue_max_cost_per_ip_24h = queue.getfloat("max_cost_per_ip_24h", config.queue_max_cost_per_ip_24h)
            config.queue_max_runtime_per_user_24h_hours = queue.getfloat("max_runtime_per_user_24h_hours", config.queue_max_runtime_per_user_24h_hours)
            config.queue_max_runtime_per_ip_24h_hours = queue.getfloat("max_runtime_per_ip_24h_hours", config.queue_max_runtime_per_ip_24h_hours)
            config.queue_max_concurrent_per_user = queue.getint("max_concurrent_per_user", config.queue_max_concurrent_per_user)
            config.queue_max_concurrent_per_ip = queue.getint("max_concurrent_per_ip", config.queue_max_concurrent_per_ip)
            config.queue_window_hours = queue.getint("window_hours", config.queue_window_hours)
            config.queue_weight_low_cost = queue.getfloat("weight_low_cost", config.queue_weight_low_cost)
            config.queue_weight_low_runtime = queue.getfloat("weight_low_runtime", config.queue_weight_low_runtime)

    return config


def get_queue_limits(config: SchedulerConfig):
    """Create QueueLimits from SchedulerConfig."""
    from .queue_order import QueueLimits
    return QueueLimits(
        max_cost_per_user_24h=config.queue_max_cost_per_user_24h,
        max_cost_per_ip_24h=config.queue_max_cost_per_ip_24h,
        max_runtime_per_user_24h_sec=int(config.queue_max_runtime_per_user_24h_hours * 3600),
        max_runtime_per_ip_24h_sec=int(config.queue_max_runtime_per_ip_24h_hours * 3600),
        max_concurrent_per_user=config.queue_max_concurrent_per_user,
        max_concurrent_per_ip=config.queue_max_concurrent_per_ip,
        weight_low_cost=config.queue_weight_low_cost,
        weight_low_runtime=config.queue_weight_low_runtime,
        window_hours=config.queue_window_hours,
    )
