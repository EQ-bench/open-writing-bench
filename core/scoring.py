import logging
import re
import statistics as stats
import random
import numpy as np
from typing import Dict, Any, List
from pathlib import Path

from utils.db_connector import db
from utils.db_schema import JudgeResult

SCORE_RANGE_MIN = 0
SCORE_RANGE_MAX = 20
def parse_judge_scores_creative(judge_model_response: str) -> Dict[str, float]:
    scores = {}

    # Parse scores using multiple regex patterns
    # Pattern 1: Metric: Score or Metric: Score X
    score_pattern1 = r'(.*?):\s*(?:Score\s+)?(-?\d+(?:\.\d+)?)'
    # Pattern 2: Metric: [Score]
    score_pattern2 = r'(.*?):\s*\[(-?\d+(?:\.\d+)?)\]'
    
    # Combine both patterns
    matches1 = re.findall(score_pattern1, judge_model_response)
    matches2 = re.findall(score_pattern2, judge_model_response)
    
    # Process matches from both patterns
    for matches in [matches1, matches2]:
        for match in matches:
            metric_name = match[0].strip()
            score = float(match[1])
            # Add check to ensure score <= 20
            if score <= SCORE_RANGE_MAX:
                scores[metric_name] = score
            # If score > 20, it's discarded/ignored

    return scores

def invert_if_negative(metric: str, score: float, negative_criteria: List[str]) -> float:
    """
    If metric is a negative criterion, invert so that higher => better:
    e.g. 20 => 0, 0 => 20 =>  new_val = 20 - old_val
    """
    if metric in negative_criteria:
        return 20.0 - score
    #print(score)
    return score


def compute_creative_scores(tasks: List[Any], negative_criteria: List[str]) -> float:
    """
    Computes average score from tasks that have aggregated_scores.
    Each task should have aggregated_scores['piece_score_0_20'] already computed.
    """
    piece_scores = []
    for task in tasks:
        # Tasks are now SQLAlchemy Task objects with aggregated_scores attribute
        if hasattr(task, 'aggregated_scores') and task.aggregated_scores:
            piece_score = task.aggregated_scores.get('piece_score_0_20')
            if piece_score is not None and isinstance(piece_score, (int, float)):
                piece_scores.append(piece_score)
        # Fallback for legacy JSON structure if needed
        elif isinstance(task, dict) and 'aggregated_scores' in task:
            piece_score = task['aggregated_scores'].get('piece_score_0_20')
            if piece_score is not None and isinstance(piece_score, (int, float)):
                piece_scores.append(piece_score)

    if not piece_scores:
        return 0.0
    return sum(piece_scores) / len(piece_scores)



def compute_single_benchmark_score_creative(tasks, negative_criteria):
    """
    Returns a dict:
      {
        "overall_score": (0..20),
        "eqbench_creative_score": (0..100)
      }
    We produce eqbench_creative_score by scaling 0..20 => 0..10 => 0..100
    """
    avg_0_20 = compute_creative_scores(tasks, negative_criteria)
    # scale to 0..100    
    eqbench_score = avg_0_20 * 5.0
    eqbench_score = round(eqbench_score, 2)
    return {
        "overall_score": round(avg_0_20, 2),
        "eqbench_creative_score": eqbench_score
    }


def bootstrap_benchmark_stability_creative(tasks, negative_criteria, n_bootstrap=500, confidence_level=0.95):
    """
    Bootstraps the final overall_score from a sample of tasks. Return a dict with stats.
    """
    original_result = compute_single_benchmark_score_creative(tasks, negative_criteria)
    original_score = original_result["overall_score"]
    if not tasks:
        return {
            "error": "No tasks found for bootstrap"
        }

    # We'll treat each entire "task" as a sampling unit
    boot_scores = []
    for _ in range(n_bootstrap):
        sample_tasks = random.choices(tasks, k=len(tasks))
        sc = compute_single_benchmark_score_creative(sample_tasks, negative_criteria)
        boot_scores.append(sc["overall_score"])

    boot_scores.sort()
    lower_idx = int((1 - confidence_level)/2 * len(boot_scores))
    upper_idx = int((1 + confidence_level)/2 * len(boot_scores)) - 1
    lower_idx = max(0, lower_idx)
    upper_idx = min(upper_idx, len(boot_scores)-1)

    ci_lower = boot_scores[lower_idx]
    ci_upper = boot_scores[upper_idx]
    mean_ = np.mean(boot_scores)
    std_ = np.std(boot_scores, ddof=1)

    return {
        "original": original_score,
        "bootstrap_mean": float(mean_),
        "bootstrap_std": float(std_),
        "standard_error": float(std_),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "confidence_level": confidence_level,
        "n_bootstrap": n_bootstrap
    }


def aggregate_ensemble_scores(task_id: int, aggregation_method: str = 'average_with_outlier_removal'):
    """
    Aggregates scores from multiple judges for a single task into a final score.
    
    Reads all JudgeResult rows for the task, computes per-metric averages across judges,
    applies invert_if_negative for negative criteria, and saves the aggregated score
    to Task.aggregated_scores.
    
    Args:
        task_id: The database ID of the task
        aggregation_method: Method for aggregation ('average' or 'average_with_outlier_removal')
    """
    with db.get_session() as session:
        # Get all judge results for this task, ordered by judge_order_index
        judge_results = session.query(JudgeResult).filter_by(task_id=task_id).order_by(JudgeResult.judge_order_index).all()
        
        if not judge_results:
            logging.warning(f"No judge results found for task {task_id}")
            return
        
        # Get the task to access negative_criteria if needed (stored in run_config)
        from utils.db_schema import Task
        task = session.query(Task).filter_by(id=task_id).first()
        if not task:
            logging.error(f"Task {task_id} not found")
            return
        
        # Collect all metrics and their scores from each judge
        metric_scores = {}  # metric_name -> [score1, score2, ...]
        
        for jr in judge_results:
            if jr.judge_scores and not jr.judge_scores.get('error'):
                for metric, score in jr.judge_scores.items():
                    if isinstance(score, (int, float)) and score <= SCORE_RANGE_MAX:
                        if metric not in metric_scores:
                            metric_scores[metric] = []
                        metric_scores[metric].append(score)
        
        if not metric_scores:
            logging.warning(f"No valid scores found for task {task_id}")
            db.update_task(task_id, {"status": "error", "error_message": "No valid judge scores"})
            return
        
        # Aggregate scores per metric
        aggregated_metrics = {}
        for metric, scores in metric_scores.items():
            if aggregation_method == 'average_with_outlier_removal' and len(scores) >= 3:
                # Remove min and max outliers if we have at least 3 judges
                sorted_scores = sorted(scores)
                trimmed = sorted_scores[1:-1]  # Remove lowest and highest
                avg_score = sum(trimmed) / len(trimmed)
            else:
                # Simple average
                avg_score = sum(scores) / len(scores)
            
            aggregated_metrics[metric] = round(avg_score, 2)
        
        # Load negative criteria from somewhere accessible
        # For now, we'll assume it's passed or stored; let's load from a known location
        # Actually, the negative_criteria should be available from the benchmark logic
        # For simplicity, we'll apply invert_if_negative with an empty list for now
        # The caller (compute_benchmark_results_creative) has access to it
        # So we'll store raw scores here and let the compute function handle inversion
        
        # Compute overall piece score (0-20) by averaging all metric scores
        # We need to know which metrics are negative to invert them
        # Let's read negative_criteria from a standard location
        from pathlib import Path
        neg_path = None
        try:
            neg_path = (task.run.run_config or {}).get("negative_criteria_file")
        except Exception:
            neg_path = None

        try:
            path_to_read = Path(neg_path) if neg_path else Path("data/negative_criteria.txt")
            negative_criteria = [line.strip() for line in path_to_read.read_text(encoding='utf-8').splitlines() if line.strip()]
        except Exception:
            negative_criteria = []
            logging.warning(f"Could not load negative criteria for task {task_id} aggregation from '{neg_path or 'data/negative_criteria.txt'}'")

        
        # Apply inversion and compute average
        inverted_scores = []
        for metric, score in aggregated_metrics.items():
            inverted = invert_if_negative(metric, score, negative_criteria)
            if inverted <= SCORE_RANGE_MAX:
                inverted_scores.append(inverted)
        
        if not inverted_scores:
            logging.warning(f"No valid inverted scores for task {task_id}")
            piece_score = 0.0
        else:
            piece_score = round(sum(inverted_scores) / len(inverted_scores), 2)
        
        # Save aggregated scores to task
        aggregated_data = {
            "piece_score_0_20": piece_score,
            "per_metric": aggregated_metrics,
            "n_judges": len(judge_results)
        }
        
        db.update_task(task_id, {
            "aggregated_scores": aggregated_data,
            "status": "completed"
        })
        
        logging.debug(f"Aggregated scores for task {task_id}: {piece_score:.2f} from {len(judge_results)} judges")

def aggregate_ensemble_scores_bulk(run_key: str, aggregation_method: str = 'average_with_outlier_removal'):
    """
    Bulk-aggregate all 'judged' tasks for a run in one session:
      - pull all judge rows once
      - compute per-task aggregates in memory
      - bulk update Task rows
    """
    from utils.db_schema import Task  # local import to avoid cycles
    with db.get_session() as session:
        # tasks to aggregate
        tasks_q = (
            session.query(Task.id, Task.run_key)
            .filter(Task.run_key == run_key, Task.status == 'judged')
        )
        task_rows = tasks_q.all()
        if not task_rows:
            logging.info(f"No 'judged' tasks to aggregate for run {run_key}")
            return

        task_ids = [t.id for t in task_rows]

        # load all judge results in one query
        jrs = (
            session.query(JudgeResult)
            .filter(JudgeResult.task_id.in_(task_ids))
            .order_by(JudgeResult.task_id, JudgeResult.judge_order_index)
            .all()
        )

        # load negative criteria once
        from utils.db_schema import Run
        run = session.query(Run).filter_by(run_key=run_key).first()
        neg_path = None
        if run and run.run_config:
            neg_path = (run.run_config or {}).get("negative_criteria_file")
        try:
            path_to_read = Path(neg_path) if neg_path else Path("data/negative_criteria.txt")
            negative_criteria = [line.strip() for line in path_to_read.read_text(encoding='utf-8').splitlines() if line.strip()]
        except Exception:
            negative_criteria = []
            logging.warning(f"Could not load negative criteria for run {run_key} from '{neg_path or 'data/negative_criteria.txt'}'")

        # group results by task_id
        by_task: Dict[int, List[JudgeResult]] = {}
        for jr in jrs:
            by_task.setdefault(jr.task_id, []).append(jr)

        updates = []
        for task_id in task_ids:
            jr_list = by_task.get(task_id, [])
            if not jr_list:
                # no judges → leave task as-is
                continue

            # collect metric scores across judges
            metric_scores: Dict[str, List[float]] = {}
            for jr in jr_list:
                if jr.judge_scores and not jr.judge_scores.get('error'):
                    for metric, score in jr.judge_scores.items():
                        if isinstance(score, (int, float)) and score <= SCORE_RANGE_MAX:
                            metric_scores.setdefault(metric, []).append(score)

            if not metric_scores:
                # mark error
                updates.append({"id": task_id, "status": "error", "aggregated_scores": {"error": "No valid judge scores"}})
                continue

            # aggregate per metric
            aggregated_metrics: Dict[str, float] = {}
            for metric, scores in metric_scores.items():
                if aggregation_method == 'average_with_outlier_removal' and len(scores) >= 3:
                    s = sorted(scores)
                    s = s[1:-1]  # drop min/max
                    avg = sum(s) / len(s)
                else:
                    avg = sum(scores) / len(scores)
                aggregated_metrics[metric] = round(avg, 2)

            # invert negatives and compute piece score
            inverted_scores = []
            for metric, score in aggregated_metrics.items():
                inv = invert_if_negative(metric, score, negative_criteria)
                if inv <= SCORE_RANGE_MAX:
                    inverted_scores.append(inv)
            piece_score = round(sum(inverted_scores) / len(inverted_scores), 2) if inverted_scores else 0.0

            aggregated_data = {
                "piece_score_0_20": piece_score,
                "per_metric": aggregated_metrics,
                "n_judges": len(jr_list),
            }
            updates.append({"id": task_id, "aggregated_scores": aggregated_data, "status": "completed"})

        if updates:
            session.bulk_update_mappings(Task, updates)
            logging.info(f"Aggregated and updated {len(updates)} tasks for run {run_key}")
