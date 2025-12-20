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

    # Default job settings
    default_threads: int = 34
    default_verbosity: str = "DEBUG"
    default_test_provider: str = "vllm"
    default_judges: list[str] = None

    # Cleanup settings
    clear_hf_cache: bool = True
    kill_vllm_processes: bool = True

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

    return config
