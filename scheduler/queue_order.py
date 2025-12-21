# scheduler/queue_order.py
"""
Queue ordering logic for the submission scheduler.

This module determines the order in which pending submissions should be processed,
taking into account fairness, rate limiting, and abuse prevention.

Factors considered:
1. user_id - Primary grouping for rate limiting
2. created_ip - Secondary grouping to prevent multi-account abuse
3. Cumulative runtime within 24h window
4. Submission time (created_at) - FIFO within priority tiers
5. Total judging cost ($ spent) within 24h window
6. Number of successful completions within 24h window

The system uses:
- Hard caps: Jobs that exceed limits are HELD (not processed until window expires)
- Soft scoring: Jobs within limits are ordered by priority score
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional
import json


@dataclass
class SubmissionData:
    """Data for a single submission, used for queue ordering."""
    id: str
    user_id: Optional[str]
    created_ip: Optional[str]  # Hashed IP
    created_at: datetime
    status: str

    # From related run data (if exists)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    results: Optional[dict] = None  # Contains judging_costs.total_judging_cost_usd

    # User info
    user_role: Optional[str] = None  # "admin" users bypass limits


@dataclass
class QueueLimits:
    """Configurable limits for queue ordering."""
    # Hard caps - jobs exceeding these are HELD
    max_cost_per_user_24h: float = 10.0  # $10 per user per 24h
    max_cost_per_ip_24h: float = 10.0    # $10 per IP per 24h
    max_runtime_per_user_24h_sec: int = 8 * 3600  # 8 hours runtime per user
    max_runtime_per_ip_24h_sec: int = 8 * 3600    # 8 hours per IP
    max_concurrent_per_user: int = 3  # Max queued/running jobs per user
    max_concurrent_per_ip: int = 5    # Max queued/running jobs per IP

    # Soft scoring weights (higher = more priority)
    # These affect queue ordering, not whether a job runs
    weight_low_cost: float = 10.0      # Bonus points for users who haven't spent much
    weight_low_runtime: float = 5.0    # Bonus points for users with low runtime

    # Window duration
    window_hours: int = 24


@dataclass
class UserStats:
    """Aggregated stats for a user/IP within the time window."""
    entity_id: str  # user_id or IP hash
    entity_type: str  # "user" or "ip"

    total_runtime_sec: float = 0.0
    total_cost_usd: float = 0.0
    completed_count: int = 0
    failed_count: int = 0
    queued_count: int = 0
    running_count: int = 0

    @property
    def total_jobs(self) -> int:
        return self.completed_count + self.failed_count

    @property
    def completion_ratio(self) -> float:
        if self.total_jobs == 0:
            return 1.0  # New users get benefit of the doubt
        return self.completed_count / self.total_jobs

    @property
    def concurrent_jobs(self) -> int:
        return self.queued_count + self.running_count


@dataclass
class QueueDecision:
    """The result of queue ordering for a submission."""
    submission_id: str
    action: str  # "process", "hold", "reject"
    priority_score: float = 0.0
    created_at: Optional[datetime] = None  # For FIFO tiebreaking
    hold_reason: Optional[str] = None
    hold_until: Optional[datetime] = None

    # Debug info
    user_stats: Optional[UserStats] = None
    ip_stats: Optional[UserStats] = None
    score_breakdown: dict = field(default_factory=dict)


def calculate_runtime_sec(
    started_at: Optional[datetime],
    finished_at: Optional[datetime],
    now: datetime
) -> float:
    """Calculate runtime for a job, using current time if still running."""
    if started_at is None:
        return 0.0
    end = finished_at if finished_at else now
    return max(0.0, (end - started_at).total_seconds())


def extract_cost_from_results(results: Optional[dict]) -> float:
    """Extract total judging cost from run results JSON."""
    if not results:
        return 0.0

    # Handle string JSON
    if isinstance(results, str):
        try:
            results = json.loads(results)
        except (json.JSONDecodeError, TypeError):
            return 0.0

    judging_costs = results.get("judging_costs", {})
    return judging_costs.get("total_judging_cost_usd", 0.0)


def aggregate_user_stats(
    submissions: list[SubmissionData],
    entity_id: str,
    entity_type: str,  # "user" or "ip"
    window_start: datetime,
    now: datetime
) -> UserStats:
    """Aggregate stats for a user or IP within the time window."""
    stats = UserStats(entity_id=entity_id, entity_type=entity_type)

    for sub in submissions:
        # Match by user_id or created_ip
        if entity_type == "user":
            if sub.user_id != entity_id:
                continue
        else:  # ip
            if sub.created_ip != entity_id:
                continue

        # Only count jobs within the window
        if sub.created_at < window_start:
            continue

        # Count by status
        status_lower = sub.status.lower()
        if status_lower in ("succeeded", "completed"):
            stats.completed_count += 1
            stats.total_runtime_sec += calculate_runtime_sec(sub.started_at, sub.finished_at, now)
            stats.total_cost_usd += extract_cost_from_results(sub.results)
        elif status_lower in ("failed", "timeout", "cancelled"):
            stats.failed_count += 1
            stats.total_runtime_sec += calculate_runtime_sec(sub.started_at, sub.finished_at, now)
            # Still count cost for failed jobs if they ran
            stats.total_cost_usd += extract_cost_from_results(sub.results)
        elif status_lower in ("queued", "submitted"):
            stats.queued_count += 1
        elif status_lower in ("running", "starting"):
            stats.running_count += 1
            # Running jobs contribute runtime up to now
            stats.total_runtime_sec += calculate_runtime_sec(sub.started_at, None, now)

    return stats


def calculate_entity_score(
    stats: UserStats,
    max_cost: float,
    max_runtime_sec: int,
    limits: QueueLimits
) -> tuple[float, dict]:
    """
    Calculate priority score for a single entity (user or IP).
    Higher score = higher priority.

    Returns: (score, breakdown_dict)
    """
    breakdown = {}
    score = 0.0

    # Low cost bonus - reward entities who haven't spent much
    # Inverse relationship: lower cost = higher bonus
    cost_ratio = stats.total_cost_usd / max_cost if max_cost > 0 else 0
    cost_bonus = (1.0 - min(1.0, cost_ratio)) * limits.weight_low_cost
    breakdown["low_cost"] = {"spent": round(stats.total_cost_usd, 2), "bonus": round(cost_bonus, 2)}
    score += cost_bonus

    # Low runtime bonus - reward entities with low runtime usage
    runtime_ratio = stats.total_runtime_sec / max_runtime_sec if max_runtime_sec > 0 else 0
    runtime_bonus = (1.0 - min(1.0, runtime_ratio)) * limits.weight_low_runtime
    breakdown["low_runtime"] = {"runtime_hours": round(stats.total_runtime_sec / 3600, 2), "bonus": round(runtime_bonus, 2)}
    score += runtime_bonus

    return score, breakdown


def calculate_priority_score(
    submission: SubmissionData,
    user_stats: UserStats,
    ip_stats: UserStats,
    limits: QueueLimits,
    now: datetime
) -> tuple[float, dict]:
    """
    Calculate priority score for a submission.
    Higher score = higher priority (processed first).

    Computes scores for both user and IP, uses the WORST (lowest) score.
    This prevents abuse via multiple accounts from same IP.

    Returns: (score, breakdown_dict)
    """
    # Calculate score based on user stats
    user_score, user_breakdown = calculate_entity_score(
        user_stats,
        limits.max_cost_per_user_24h,
        limits.max_runtime_per_user_24h_sec,
        limits
    )

    # Calculate score based on IP stats
    ip_score, ip_breakdown = calculate_entity_score(
        ip_stats,
        limits.max_cost_per_ip_24h,
        limits.max_runtime_per_ip_24h_sec,
        limits
    )

    # Use the WORST score (prevents multi-account abuse)
    if user_score <= ip_score:
        score = user_score
        limiting_entity = "user"
    else:
        score = ip_score
        limiting_entity = "ip"

    breakdown = {
        "user_score": round(user_score, 2),
        "user": user_breakdown,
        "ip_score": round(ip_score, 2),
        "ip": ip_breakdown,
        "limiting_entity": limiting_entity,
        "final_score": round(score, 2)
    }

    return score, breakdown


def check_hard_limits(
    user_stats: UserStats,
    ip_stats: UserStats,
    limits: QueueLimits,
    now: datetime,
    window_start: datetime
) -> tuple[bool, Optional[str], Optional[datetime]]:
    """
    Check if submission violates hard limits.

    Returns: (should_hold, reason, hold_until)
    """
    # Check user cost limit
    if user_stats.total_cost_usd >= limits.max_cost_per_user_24h:
        return True, f"User cost limit exceeded (${user_stats.total_cost_usd:.2f} >= ${limits.max_cost_per_user_24h})", window_start + timedelta(hours=limits.window_hours)

    # Check IP cost limit
    if ip_stats.total_cost_usd >= limits.max_cost_per_ip_24h:
        return True, f"IP cost limit exceeded (${ip_stats.total_cost_usd:.2f} >= ${limits.max_cost_per_ip_24h})", window_start + timedelta(hours=limits.window_hours)

    # Check user runtime limit
    if user_stats.total_runtime_sec >= limits.max_runtime_per_user_24h_sec:
        hours = user_stats.total_runtime_sec / 3600
        limit_hours = limits.max_runtime_per_user_24h_sec / 3600
        return True, f"User runtime limit exceeded ({hours:.1f}h >= {limit_hours:.1f}h)", window_start + timedelta(hours=limits.window_hours)

    # Check IP runtime limit
    if ip_stats.total_runtime_sec >= limits.max_runtime_per_ip_24h_sec:
        hours = ip_stats.total_runtime_sec / 3600
        limit_hours = limits.max_runtime_per_ip_24h_sec / 3600
        return True, f"IP runtime limit exceeded ({hours:.1f}h >= {limit_hours:.1f}h)", window_start + timedelta(hours=limits.window_hours)

    # Check user concurrent jobs
    if user_stats.concurrent_jobs >= limits.max_concurrent_per_user:
        return True, f"User has too many concurrent jobs ({user_stats.concurrent_jobs} >= {limits.max_concurrent_per_user})", None

    # Check IP concurrent jobs
    if ip_stats.concurrent_jobs >= limits.max_concurrent_per_ip:
        return True, f"IP has too many concurrent jobs ({ip_stats.concurrent_jobs} >= {limits.max_concurrent_per_ip})", None

    return False, None, None


def order_queue(
    pending_submissions: list[SubmissionData],
    all_submissions: list[SubmissionData],
    limits: Optional[QueueLimits] = None,
    now: Optional[datetime] = None
) -> list[QueueDecision]:
    """
    Order pending submissions by priority.

    Args:
        pending_submissions: Submissions in SUBMITTED/QUEUED status to be ordered
        all_submissions: All submissions (including completed) for stats calculation
        limits: Queue limits configuration
        now: Current time (for testing)

    Returns:
        List of QueueDecision objects, sorted by priority (highest first)
    """
    if limits is None:
        limits = QueueLimits()
    if now is None:
        now = datetime.now(timezone.utc)

    window_start = now - timedelta(hours=limits.window_hours)

    # Pre-compute stats for all users and IPs
    user_stats_cache: dict[str, UserStats] = {}
    ip_stats_cache: dict[str, UserStats] = {}

    # Get unique user_ids and IPs from pending submissions
    unique_users = {s.user_id for s in pending_submissions if s.user_id}
    unique_ips = {s.created_ip for s in pending_submissions if s.created_ip}

    for user_id in unique_users:
        user_stats_cache[user_id] = aggregate_user_stats(
            all_submissions, user_id, "user", window_start, now
        )

    for ip in unique_ips:
        ip_stats_cache[ip] = aggregate_user_stats(
            all_submissions, ip, "ip", window_start, now
        )

    # Max possible score for admin priority boost
    max_score = limits.weight_low_cost + limits.weight_low_runtime

    decisions: list[QueueDecision] = []

    for sub in pending_submissions:
        # Admin users bypass all limits and get max priority
        is_admin = sub.user_role == "admin"

        if is_admin:
            # Admin users bypass limits but get normal max score (no boost)
            decisions.append(QueueDecision(
                submission_id=sub.id,
                action="process",
                priority_score=max_score,
                created_at=sub.created_at,
                user_stats=None,
                ip_stats=None,
                score_breakdown={"admin": True, "final_score": max_score}
            ))
            continue

        # Get stats (or create empty stats for anonymous)
        user_stats = user_stats_cache.get(sub.user_id) or UserStats(entity_id="anonymous", entity_type="user")
        ip_stats = ip_stats_cache.get(sub.created_ip) or UserStats(entity_id="unknown", entity_type="ip")

        # Check hard limits
        should_hold, hold_reason, hold_until = check_hard_limits(
            user_stats, ip_stats, limits, now, window_start
        )

        if should_hold:
            decisions.append(QueueDecision(
                submission_id=sub.id,
                action="hold",
                priority_score=0.0,
                created_at=sub.created_at,
                hold_reason=hold_reason,
                hold_until=hold_until,
                user_stats=user_stats,
                ip_stats=ip_stats
            ))
        else:
            # Calculate priority score
            score, breakdown = calculate_priority_score(
                sub, user_stats, ip_stats, limits, now
            )

            decisions.append(QueueDecision(
                submission_id=sub.id,
                action="process",
                priority_score=score,
                created_at=sub.created_at,
                user_stats=user_stats,
                ip_stats=ip_stats,
                score_breakdown=breakdown
            ))

    # Sort by:
    # 1. action (process before hold)
    # 2. priority score (descending - higher priority first)
    # 3. created_at (ascending - FIFO for same priority)
    decisions.sort(key=lambda d: (
        0 if d.action == "process" else 1,
        -d.priority_score,
        d.created_at or datetime.min.replace(tzinfo=timezone.utc)
    ))

    return decisions


def get_next_submission_id(
    pending_submissions: list[SubmissionData],
    all_submissions: list[SubmissionData],
    limits: Optional[QueueLimits] = None,
    now: Optional[datetime] = None
) -> Optional[str]:
    """
    Convenience function to get the ID of the next submission to process.

    Returns None if all pending submissions are held.
    """
    decisions = order_queue(pending_submissions, all_submissions, limits, now)

    for decision in decisions:
        if decision.action == "process":
            return decision.submission_id

    return None
