# scheduler/test_queue_order_detailed.py
"""
Detailed test showing queue ordering effects with a larger dataset.

Run with: python -m scheduler.test_queue_order_detailed
"""

from datetime import datetime, timezone, timedelta
from .queue_order import (
    SubmissionData, QueueLimits,
    order_queue
)


def make_submission(
    id: str,
    user_id: str,
    created_ip: str,
    created_at: datetime,
    status: str = "SUBMITTED",
    started_at: datetime = None,
    finished_at: datetime = None,
    cost_usd: float = 0.0,
    user_role: str = None
) -> SubmissionData:
    """Helper to create test submission data."""
    results = {"judging_costs": {"total_judging_cost_usd": cost_usd}} if cost_usd else None
    return SubmissionData(
        id=id,
        user_id=user_id,
        created_ip=created_ip,
        created_at=created_at,
        status=status,
        started_at=started_at,
        finished_at=finished_at,
        results=results,
        user_role=user_role
    )


def run_comparison():
    """Run a detailed comparison showing weighting effects."""
    print("=" * 100)
    print("QUEUE ORDERING - DETAILED COMPARISON")
    print("=" * 100)

    now = datetime.now(timezone.utc)

    # Create a diverse set of historical submissions
    history = []

    # === User histories ===

    # whale: Heavy user - 3 completed jobs, $9 spent, 6 hours runtime
    for i in range(3):
        history.append(make_submission(
            f"whale_h{i}", "whale", "ip_whale",
            now - timedelta(hours=20 - i*4),
            status="SUCCEEDED", cost_usd=3.0,
            started_at=now - timedelta(hours=21 - i*4),
            finished_at=now - timedelta(hours=19 - i*4)
        ))

    # medium1: Medium user - 2 jobs, $4 spent, 3 hours runtime
    for i in range(2):
        history.append(make_submission(
            f"medium1_h{i}", "medium1", "ip_medium1",
            now - timedelta(hours=18 - i*6),
            status="SUCCEEDED", cost_usd=2.0,
            started_at=now - timedelta(hours=19 - i*6),
            finished_at=now - timedelta(hours=17.5 - i*6)
        ))

    # medium2: Medium user - 1 job, $5 spent, 2 hours runtime
    history.append(make_submission(
        "medium2_h0", "medium2", "ip_medium2",
        now - timedelta(hours=10),
        status="SUCCEEDED", cost_usd=5.0,
        started_at=now - timedelta(hours=12),
        finished_at=now - timedelta(hours=10)
    ))

    # light1: Light user - 1 job, $1 spent, 1 hour runtime
    history.append(make_submission(
        "light1_h0", "light1", "ip_light1",
        now - timedelta(hours=15),
        status="SUCCEEDED", cost_usd=1.0,
        started_at=now - timedelta(hours=16),
        finished_at=now - timedelta(hours=15)
    ))

    # light2: Light user - 1 job, $0.50 spent, 0.5 hour runtime
    history.append(make_submission(
        "light2_h0", "light2", "ip_light2",
        now - timedelta(hours=22),
        status="SUCCEEDED", cost_usd=0.5,
        started_at=now - timedelta(hours=22.5),
        finished_at=now - timedelta(hours=22)
    ))

    # new1, new2, new3: Brand new users with no history

    # shared_ip_user1, shared_ip_user2: Two different users from same IP
    # The IP has $6 total spent across both users
    history.append(make_submission(
        "shared1_h0", "shared_ip_user1", "shared_ip",
        now - timedelta(hours=8),
        status="SUCCEEDED", cost_usd=3.0,
        started_at=now - timedelta(hours=9),
        finished_at=now - timedelta(hours=8)
    ))
    history.append(make_submission(
        "shared2_h0", "shared_ip_user2", "shared_ip",
        now - timedelta(hours=6),
        status="SUCCEEDED", cost_usd=3.0,
        started_at=now - timedelta(hours=7),
        finished_at=now - timedelta(hours=6)
    ))

    # === Pending submissions (the queue) ===
    # Vary submission times to see FIFO effects
    pending = [
        make_submission("p_whale", "whale", "ip_whale", now - timedelta(minutes=30)),
        make_submission("p_medium1", "medium1", "ip_medium1", now - timedelta(minutes=90)),
        make_submission("p_medium2", "medium2", "ip_medium2", now - timedelta(minutes=60)),
        make_submission("p_light1", "light1", "ip_light1", now - timedelta(minutes=120)),
        make_submission("p_light2", "light2", "ip_light2", now - timedelta(minutes=45)),
        make_submission("p_new1", "new1", "ip_new1", now - timedelta(minutes=15)),
        make_submission("p_new2", "new2", "ip_new2", now - timedelta(minutes=75)),
        make_submission("p_new3", "new3", "ip_new3", now - timedelta(minutes=105)),
        make_submission("p_shared1", "shared_ip_user1", "shared_ip", now - timedelta(minutes=40)),
        make_submission("p_shared2", "shared_ip_user2", "shared_ip", now - timedelta(minutes=55)),
        # New user but using the shared abusive IP
        make_submission("p_new_shared", "new_on_shared_ip", "shared_ip", now - timedelta(minutes=20)),
        # Admin user - should bypass all limits and get top priority
        make_submission("p_admin", "admin_user", "ip_admin", now - timedelta(minutes=5), user_role="admin"),
    ]

    # Build a lookup for user info
    user_info = {}
    for sub in history:
        uid = sub.user_id
        if uid not in user_info:
            user_info[uid] = {"cost": 0.0, "runtime_h": 0.0, "jobs": 0}
        user_info[uid]["cost"] += sub.results.get("judging_costs", {}).get("total_judging_cost_usd", 0) if sub.results else 0
        if sub.started_at and sub.finished_at:
            user_info[uid]["runtime_h"] += (sub.finished_at - sub.started_at).total_seconds() / 3600
        user_info[uid]["jobs"] += 1

    # Add new users
    for uid in ["new1", "new2", "new3", "new_on_shared_ip", "admin_user"]:
        user_info[uid] = {"cost": 0.0, "runtime_h": 0.0, "jobs": 0}

    # IP info
    ip_info = {}
    for sub in history:
        ip = sub.created_ip
        if ip not in ip_info:
            ip_info[ip] = {"cost": 0.0, "runtime_h": 0.0}
        ip_info[ip]["cost"] += sub.results.get("judging_costs", {}).get("total_judging_cost_usd", 0) if sub.results else 0
        if sub.started_at and sub.finished_at:
            ip_info[ip]["runtime_h"] += (sub.finished_at - sub.started_at).total_seconds() / 3600

    print("\n" + "=" * 100)
    print("USER HISTORY (within 24h window)")
    print("=" * 100)
    print(f"{'User':<20} {'IP':<15} {'Jobs':>5} {'Cost':>8} {'Runtime':>10}")
    print("-" * 60)
    for p in pending:
        uid = p.user_id
        ip = p.created_ip
        info = user_info.get(uid, {"cost": 0, "runtime_h": 0, "jobs": 0})
        ip_i = ip_info.get(ip, {"cost": 0, "runtime_h": 0})
        print(f"{uid:<20} {ip:<15} {info['jobs']:>5} ${info['cost']:>7.2f} {info['runtime_h']:>9.1f}h")
    print()
    print("IP totals:")
    for ip, info in sorted(ip_info.items(), key=lambda x: -x[1]["cost"]):
        print(f"  {ip}: ${info['cost']:.2f}, {info['runtime_h']:.1f}h runtime")

    # === Compare: Pure FIFO vs Weighted ===

    print("\n" + "=" * 100)
    print("COMPARISON: PURE FIFO vs WEIGHTED ORDERING")
    print("=" * 100)

    # Pure FIFO (no weighting - just sort by created_at)
    fifo_order = sorted(pending, key=lambda s: s.created_at)

    # Weighted ordering with default limits
    limits = QueueLimits()
    weighted_decisions = order_queue(pending, history + pending, limits=limits, now=now)

    # Print side by side
    print(f"\n{'PURE FIFO (by submission time)':<50} | {'WEIGHTED (by usage + FIFO tiebreak)':<50}")
    print("-" * 102)

    max_len = max(len(fifo_order), len(weighted_decisions))
    for i in range(max_len):
        # FIFO side
        if i < len(fifo_order):
            f = fifo_order[i]
            uid = f.user_id
            wait_min = (now - f.created_at).total_seconds() / 60
            fifo_str = f"{i+1:>2}. {uid:<18} (waited {wait_min:>3.0f}m)"
        else:
            fifo_str = ""

        # Weighted side
        if i < len(weighted_decisions):
            w = weighted_decisions[i]
            uid = next((p.user_id for p in pending if p.id == w.submission_id), "?")
            wait_min = (now - w.created_at).total_seconds() / 60 if w.created_at else 0
            if w.action == "hold":
                weighted_str = f"{i+1:>2}. {uid:<18} HELD: {w.hold_reason[:20]}..."
            else:
                weighted_str = f"{i+1:>2}. {uid:<18} score={w.priority_score:>5.2f} (waited {wait_min:>3.0f}m)"
        else:
            weighted_str = ""

        print(f"{fifo_str:<50} | {weighted_str:<50}")

    # Detailed breakdown
    print("\n" + "=" * 100)
    print("DETAILED SCORE BREAKDOWN (Weighted)")
    print("=" * 100)
    print(f"Limits: max_cost=${limits.max_cost_per_user_24h}/user, ${limits.max_cost_per_ip_24h}/IP | "
          f"max_runtime={limits.max_runtime_per_user_24h_sec/3600:.0f}h/user, {limits.max_runtime_per_ip_24h_sec/3600:.0f}h/IP")
    print(f"Weights: low_cost={limits.weight_low_cost}, low_runtime={limits.weight_low_runtime}")
    print(f"Max possible score: {limits.weight_low_cost + limits.weight_low_runtime:.1f}")
    print()

    print(f"{'#':<3} {'User':<20} {'Action':<8} {'Score':>6} {'Limited':>8} | "
          f"{'User$':>7} {'UserH':>6} {'UScore':>7} | {'IP$':>7} {'IPH':>6} {'IScore':>7}")
    print("-" * 110)

    for i, d in enumerate(weighted_decisions, 1):
        uid = next((p.user_id for p in pending if p.id == d.submission_id), "?")

        if d.action == "hold":
            print(f"{i:<3} {uid:<20} {'HOLD':<8} {'-':>6} {'-':>8} | "
                  f"{'-':>7} {'-':>6} {'-':>7} | {'-':>7} {'-':>6} {'-':>7}  <- {d.hold_reason}")
        else:
            sb = d.score_breakdown
            u = sb.get("user", {})
            ip = sb.get("ip", {})
            print(f"{i:<3} {uid:<20} {'process':<8} {d.priority_score:>6.2f} {sb.get('limiting_entity', '?'):>8} | "
                  f"${u.get('low_cost', {}).get('spent', 0):>6.2f} {u.get('low_runtime', {}).get('runtime_hours', 0):>5.1f}h {sb.get('user_score', 0):>7.2f} | "
                  f"${ip.get('low_cost', {}).get('spent', 0):>6.2f} {ip.get('low_runtime', {}).get('runtime_hours', 0):>5.1f}h {sb.get('ip_score', 0):>7.2f}")

    # Summary
    print("\n" + "=" * 100)
    print("OBSERVATIONS")
    print("=" * 100)

    # Find position changes
    fifo_positions = {fifo_order[i].user_id: i+1 for i in range(len(fifo_order))}
    weighted_positions = {}
    for i, d in enumerate(weighted_decisions):
        uid = next((p.user_id for p in pending if p.id == d.submission_id), None)
        if uid:
            weighted_positions[uid] = i+1

    print("\nPosition changes (FIFO -> Weighted):")
    changes = []
    for uid in fifo_positions:
        fifo_pos = fifo_positions[uid]
        weighted_pos = weighted_positions.get(uid, "HELD")
        if isinstance(weighted_pos, int):
            delta = fifo_pos - weighted_pos  # positive = moved up
            changes.append((uid, fifo_pos, weighted_pos, delta))
        else:
            changes.append((uid, fifo_pos, weighted_pos, None))

    changes.sort(key=lambda x: (x[3] is None, -(x[3] or 0)))  # Sort by improvement

    for uid, fifo_pos, weighted_pos, delta in changes:
        if delta is None:
            print(f"  {uid}: #{fifo_pos} -> HELD")
        elif delta > 0:
            print(f"  {uid}: #{fifo_pos} -> #{weighted_pos} (↑ moved up {delta} positions)")
        elif delta < 0:
            print(f"  {uid}: #{fifo_pos} -> #{weighted_pos} (↓ moved down {-delta} positions)")
        else:
            print(f"  {uid}: #{fifo_pos} -> #{weighted_pos} (no change)")

    print("\nKey effects:")
    print("  - Light users and new users get priority over heavy users")
    print("  - Users on abusive IPs (shared_ip) are penalized even if user is new")
    print("  - FIFO is preserved as tiebreaker when scores are equal")


if __name__ == "__main__":
    run_comparison()
