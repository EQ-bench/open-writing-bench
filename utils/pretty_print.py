# utils/pretty_print.py

import unicodedata
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from utils.db_schema import Run, EloRating

# ── width helpers ─────────────────────────────────────────────────────────────
def _cell_w(s: str) -> int:
    w = 0
    for ch in str(s):
        w += 2 if unicodedata.east_asian_width(ch) in ("F", "W") else 1
    return w

def _crop(s: str, max_w: int, ellipsis="…") -> str:
    s = str(s)
    if _cell_w(s) <= max_w:
        return s + " " * (max_w - _cell_w(s))
    keep_w = max_w - _cell_w(ellipsis)
    out, cur = "", 0
    for ch in s:
        ch_w = 2 if unicodedata.east_asian_width(ch) in ("F", "W") else 1
        if cur + ch_w > keep_w:
            break
        out += ch
        cur += ch_w
    return out + ellipsis + " " * (max_w - _cell_w(out + ellipsis))

# ── data helpers ──────────────────────────────────────────────────────────────
def _latest_completed_runs_by_model(session: Session) -> Dict[str, Run]:
    rows: List[Run] = (
        session.query(Run)
        .filter(Run.status == "completed")
        .all()
    )
    latest: Dict[str, Run] = {}
    for r in rows:
        key = r.test_model
        if key not in latest:
            latest[key] = r
        else:
            # prefer the one with later end_time; fall back to start_time
            a = latest[key].end_time or latest[key].start_time
            b = r.end_time or r.start_time
            if (a or datetime.min.replace(tzinfo=timezone.utc)) < (b or datetime.min.replace(tzinfo=timezone.utc)):
                latest[key] = r
    return latest

def _elo_rows(session: Session) -> List[EloRating]:
    return session.query(EloRating).all()

# ── boxes ─────────────────────────────────────────────────────────────────────
def print_summary_box_db(run: Run, elo_row: Optional[EloRating]) -> None:
    results = (run.results or {})
    bench = results.get("benchmark_results", {}) if isinstance(results, dict) else {}
    rubric_0_20 = bench.get("creative_score_0_20")
    rubric_0_100 = f"{(rubric_0_20*5.0):.2f}" if isinstance(rubric_0_20, (int, float)) else "N/A"

    elo_raw = "N/A"
    elo_norm = "N/A"
    if elo_row:
        elo_raw  = f"{(elo_row.elo if elo_row.elo is not None else 0):.2f}" if isinstance(elo_row.elo, (int, float)) else "N/A"
        elo_norm = f"{(elo_row.elo_norm if elo_row.elo_norm is not None else 0):.2f}" if isinstance(elo_row.elo_norm, (int, float)) else "N/A"

    start_time = run.start_time.isoformat() if run.start_time else "N/A"
    end_time = run.end_time.isoformat() if run.end_time else "N/A"

    # box
    BOX_W = 80
    LINE  = "─"
    TOP   = "┌" + LINE * (BOX_W - 2) + "┐"
    BOT   = "└" + LINE * (BOX_W - 2) + "┘"
    SEP   = "├" + LINE * (BOX_W - 2) + "┤"
    TSEP  = "╞" + LINE * (BOX_W - 2) + "╡"
    PAD   = 1

    rows = [
        ("Run Key:", run.run_key),
        ("Model:", run.test_model),
        ("Status:", run.status),
        ("Start:", start_time),
        ("End:", end_time),
        ("Rubric (0-100):", rubric_0_100),
        ("ELO Raw:", elo_raw),
        ("ELO Norm:", elo_norm),
    ]

    # compute col widths
    label_col = min(max(_cell_w(lbl) for lbl, _ in rows), BOX_W - (3 + 4*PAD) - 15)
    value_col = BOX_W - (3 + 4*PAD) - label_col

    def _row(lbl: str, val: str) -> str:
        lbf = _crop(lbl, label_col)
        vbf = _crop(val, value_col)
        return f"│{' '*PAD}{lbf}{' '*PAD}│{' '*PAD}{vbf}{' '*PAD}│"

    print("\n" + TOP)
    print(f"│{('Open Writing Bench Results').center(BOX_W-2)}│")
    print(TSEP)
    for i, (lbl, val) in enumerate(rows):
        if i in (2, 5):
            print(SEP)
        print(_row(lbl, str(val)))
    print(BOT)

# ── leaderboards (centered window) ────────────────────────────────────────────
def print_elo_leaderboard_window(elo_rows: List[EloRating], highlight_model: str, window_size: int = 10) -> None:
    # prepare
    entries = []
    for r in elo_rows:
        ci = None
        if isinstance(r.ci_low_norm, (int, float)) and isinstance(r.ci_high_norm, (int, float)):
            ci = f"{r.ci_low_norm:.0f} - {r.ci_high_norm:.0f}"
        entries.append({
            "name": r.model_name,
            "elo_norm": r.elo_norm if isinstance(r.elo_norm, (int, float)) else float("-inf"),
            "elo_norm_disp": f"{r.elo_norm:.0f}" if isinstance(r.elo_norm, (int, float)) else "N/A",
            "elo_raw_disp": f"{r.elo:.0f}" if isinstance(r.elo, (int, float)) else "N/A",
            "sigma_disp": f"{r.sigma:.1f}" if isinstance(r.sigma, (int, float)) else "N/A",
            "ci_disp": ci or "N/A",
        })
    entries.sort(key=lambda x: x["elo_norm"], reverse=True)

    # center on highlight
    idx = next((i for i, e in enumerate(entries) if e["name"] == highlight_model), None)
    if idx is None:
        idx = 0
    start = max(0, idx - window_size)
    end   = min(len(entries), start + (2*window_size + 1))
    start = max(0, end - (2*window_size + 1))
    window = entries[start:end]

    # table layout
    COL_PAD = 1
    RANK_W, MODEL_W, EN_W, ER_W, CI_W, SIG_W = 4, 35, 8, 8, 12, 7
    cols = [RANK_W, MODEL_W, EN_W, ER_W, CI_W, SIG_W]
    LINE = "─"
    segs = [LINE * (w + 2*COL_PAD) for w in cols]
    ROW_SEP = f"├{'┼'.join(segs)}┤"
    BOX_W = len(ROW_SEP)
    TOP   = "┌" + LINE * (BOX_W - 2) + "┐"
    BOT   = "└" + LINE * (BOX_W - 2) + "┘"
    TSEP  = "╞" + LINE * (BOX_W - 2) + "╡"

    headers = ["Rank", "Model Name", "ELO Norm", "ELO Raw", "95% CI Norm", "Sigma"]
    hdr_cells = [
        _crop(h, w).center(w) for h, w in zip(headers, [RANK_W, MODEL_W, EN_W, ER_W, CI_W, SIG_W])
    ]
    hdr_row = "│" + "│".join([f"{' '*COL_PAD}{c}{' '*COL_PAD}" for c in hdr_cells]) + "│"

    print("\n" + TOP)
    print(f"│{('ELO Leaderboard (centered on ' + highlight_model + ')').center(BOX_W-2)}│")
    print(TSEP)
    print(hdr_row)
    print(ROW_SEP)

    for i, e in enumerate(window, start=start+1):
        pref = ">" if e["name"] == highlight_model else " "
        r_str = _crop(f"{pref}{i}", RANK_W)
        row_cells = [
            r_str,
            _crop(e["name"], MODEL_W),
            _crop(e["elo_norm_disp"], EN_W).rjust(EN_W),
            _crop(e["elo_raw_disp"], ER_W).rjust(ER_W),
            _crop(e["ci_disp"], CI_W).center(CI_W),
            _crop(e["sigma_disp"], SIG_W).rjust(SIG_W),
        ]
        row = "│" + "│".join([f"{' '*COL_PAD}{c}{' '*COL_PAD}" for c in row_cells]) + "│"
        print(row)
    print(BOT)

def print_rubric_leaderboard_window(latest_runs_by_model: Dict[str, Run], highlight_model: str, window_size: int = 10) -> None:
    # build entries
    rows = []
    for model, run in latest_runs_by_model.items():
        res = (run.results or {})
        bench = res.get("benchmark_results", {}) if isinstance(res, dict) else {}
        sc0_20 = bench.get("creative_score_0_20")
        if isinstance(sc0_20, (int, float)):
            score100 = sc0_20 * 5.0
            rows.append({"name": model, "score": score100, "disp": f"{score100:.1f}"})

    if not rows:
        print("\n[INFO] No rubric data to display.")
        return

    rows.sort(key=lambda x: x["score"], reverse=True)
    idx = next((i for i, e in enumerate(rows) if e["name"] == highlight_model), None)
    if idx is None:
        idx = 0
    start = max(0, idx - window_size)
    end   = min(len(rows), start + (2*window_size + 1))
    start = max(0, end - (2*window_size + 1))
    window = rows[start:end]

    COL_PAD = 1
    RANK_W, MODEL_W, SCORE_W = 4, 50, 15
    cols = [RANK_W, MODEL_W, SCORE_W]
    LINE = "─"
    segs = [LINE * (w + 2*COL_PAD) for w in cols]
    ROW_SEP = f"├{'┼'.join(segs)}┤"
    BOX_W = len(ROW_SEP)
    TOP   = "┌" + LINE * (BOX_W - 2) + "┐"
    BOT   = "└" + LINE * (BOX_W - 2) + "┘"
    TSEP  = "╞" + LINE * (BOX_W - 2) + "╡"

    headers = ["Rank", "Model Name", "Rubric (0-100)"]
    hdr_cells = [_crop(h, w).center(w) for h, w in zip(headers, [RANK_W, MODEL_W, SCORE_W])]
    hdr_row = "│" + "│".join([f"{' '*COL_PAD}{c}{' '*COL_PAD}" for c in hdr_cells]) + "│"

    print("\n" + TOP)
    print(f"│{('Rubric Leaderboard (centered on ' + highlight_model + ')').center(BOX_W-2)}│")
    print(TSEP)
    print(hdr_row)
    print(ROW_SEP)

    for i, e in enumerate(window, start=start+1):
        pref = ">" if e["name"] == highlight_model else " "
        r_str = _crop(f"{pref}{i}", RANK_W)
        cells = [
            r_str,
            _crop(e["name"], MODEL_W),
            _crop(e["disp"], SCORE_W).rjust(SCORE_W),
        ]
        row = "│" + "│".join([f"{' '*COL_PAD}{c}{' '*COL_PAD}" for c in cells]) + "│"
        print(row)
    print(BOT)

# ── one-shot entry ────────────────────────────────────────────────────────────
def print_postrun_displays(session: Session, run_key: str) -> None:
    run = session.query(Run).filter_by(run_key=run_key).first()
    if not run:
        print(f"\n[INFO] Run {run_key} not found.")
        return

    elo = None
    try:
        elo = session.query(EloRating).filter_by(model_name=run.test_model).first()
    except Exception:
        elo = None

    print_summary_box_db(run, elo)

    # rubric window
    latest = _latest_completed_runs_by_model(session)
    print_rubric_leaderboard_window(latest, run.test_model, window_size=10)

    # elo window
    try:
        elo_rows = _elo_rows(session)
    except Exception:
        elo_rows = []
    if elo_rows:
        print_elo_leaderboard_window(elo_rows, run.test_model, window_size=10)
