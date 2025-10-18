# core/benchmark.py

"""
Core orchestration logic for the Creative Writing Benchmark.

This module contains the main `run_eq_bench_creative` function which manages
the entire lifecycle of a benchmark run using a database backend. It handles
run initialization, task creation, parallelized generation and judging,
final scoring, and ELO analysis.
"""

import os
import re
import uuid
import time
import logging
from datetime import datetime
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from typing import Dict, List, Optional, Any

from utils.db_connector import db
from utils.db_schema import Run, Task
from utils.api import get_client
from core.conversation import CreativeWritingTask
from core.scoring import (
    compute_single_benchmark_score_creative,
    bootstrap_benchmark_stability_creative,
    aggregate_ensemble_scores
)
from core.elo import run_elo_analysis_creative

def compute_benchmark_results_creative(run_key: str, negative_criteria: List[str]):
    """
    Gathers all completed tasks from the DB for the run, aggregates their final
    scores, performs bootstrap analysis, and saves the results to the run record.
    """
    logging.info(f"Aggregating ensemble scores for run {run_key}...")
    tasks_to_aggregate = db.get_tasks_for_run(run_key, status_filter='judged')
    for task in tqdm(tasks_to_aggregate, desc="Aggregating Scores"):
        # This function calculates the final score from the ensemble and saves it
        # to the task record, changing its status to 'completed'.
        aggregate_ensemble_scores(task.id, aggregation_method='average_with_outlier_removal')

    logging.info(f"Calculating final benchmark results for run {run_key}...")
    completed_tasks = db.get_tasks_for_run(run_key, status_filter='completed')

    if not completed_tasks:
        logging.warning(f"No completed tasks with aggregated scores found for run {run_key}.")
        run_updates = {"results": {"benchmark_results": {"error": "No completed tasks with scores"}}}
        db.update_run(run_key, run_updates)
        return

    summary_result = compute_single_benchmark_score_creative(completed_tasks, negative_criteria)
    boot_stats = bootstrap_benchmark_stability_creative(completed_tasks, negative_criteria)

    # Prepare final results structure
    current_run_data = db.get_run(run_key)
    results_dict = current_run_data.results or {}
    bench_results = results_dict.get("benchmark_results", {})

    bench_results.update({
        "creative_score_0_20": summary_result["overall_score"],
        "eqbench_creative_score": summary_result["eqbench_creative_score"],
        "bootstrap_analysis": boot_stats
    })
    results_dict["benchmark_results"] = bench_results
    
    db.update_run(run_key, {"results": results_dict})

    logging.info(f"Creative benchmark summary => Score(0-100)={summary_result['eqbench_creative_score']:.2f}")
    if "error" not in boot_stats:
        logging.info(f"Bootstrap 95% CI: ({boot_stats['ci_lower']:.2f}, {boot_stats['ci_upper']:.2f})")


def run_eq_bench_creative(
    test_model: str,
    judge_models: List[str],
    num_threads: int,
    run_id: Optional[str],
    creative_prompts_file: str,
    creative_criteria_file: str,
    negative_criteria_file: str,
    judge_prompt_file: str,
    redo_judging: bool,
    iterations: int,
    run_elo: bool,
    vllm_params_file: Optional[str]
) -> str:
    """
    Main function to run the creative writing benchmark using the database.
    """
    # --- 1. Initialize Run and Load Assets ---
    sanitized_model = re.sub(r'[^a-zA-Z0-9_-]+', '_', test_model)
    base_id = run_id if run_id else str(uuid.uuid4())
    run_key = f"{base_id}__{sanitized_model}"

    run_config = {
        "judge_models": judge_models,
        "iterations": iterations,
        "creative_prompts_file": creative_prompts_file,
        "vllm_params_file": vllm_params_file
    }
    db.get_or_create_run(run_key, test_model, run_config)

    # Load criteria and prompts from files (original logic)
    creative_writing_criteria = [line.strip() for line in Path(creative_criteria_file).read_text(encoding='utf-8').splitlines() if line.strip()]
    negative_criteria = [line.strip() for line in Path(negative_criteria_file).read_text(encoding='utf-8').splitlines() if line.strip()]
    judge_prompt_template = Path(judge_prompt_file).read_text(encoding='utf-8')
    with open(creative_prompts_file, 'r', encoding='utf-8') as f:
        creative_prompts = json.load(f)

    # --- 2. Prepare Tasks ---
    logging.info("Preparing tasks...")
    existing_tasks_map = {f"{t.prompt_id}_{t.iteration_index}": t for t in db.get_tasks_for_run(run_key)}
    tasks_to_create = []

    for prompt_key, prompt_obj in creative_prompts.items():
        for i in range(1, iterations + 1):
            task_key = f"{prompt_key}_{i}"
            if task_key not in existing_tasks_map:
                tasks_to_create.append(Task(
                    run_key=run_key,
                    prompt_id=prompt_key,
                    iteration_index=i,
                    status='initialized'
                ))
            elif redo_judging and existing_tasks_map[task_key].status in ['judged', 'completed']:
                logging.info(f"Marking task for prompt {prompt_key} iter {i} for re-judging.")
                db.reset_judging_for_task(existing_tasks_map[task_key].id) # Assumes new DB connector function

    if tasks_to_create:
        logging.info(f"Creating {len(tasks_to_create)} new tasks in the database.")
        db.bulk_insert_tasks(tasks_to_create)

    # --- 3. Generation Phase ---
    logging.info("Starting generation phase...")
    tasks_to_generate = db.get_tasks_for_run(run_key, status_filter='initialized')
    if tasks_to_generate:
        test_model_client = get_client(test_model, client_type='test', vllm_params_file=vllm_params_file)
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for task in tasks_to_generate:
                prompt_obj = creative_prompts[task.prompt_id]
                base_prompt = prompt_obj["writing_prompt"]
                seed_mods = prompt_obj["seed_modifiers"]
                seed_modifier = seed_mods[(task.iteration_index - 1) % len(seed_mods)]
                
                task_controller = CreativeWritingTask(task)
                futures.append(executor.submit(task_controller.generate_creative_piece, test_model_client, base_prompt, seed_modifier))

            for future in tqdm(list(futures), desc="Generating creative pieces"):
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"An error occurred during generation future execution: {e}", exc_info=True)
    else:
        logging.info("No tasks require generation.")

    # --- 4. Judging Phase ---
    logging.info("Starting judging phase...")
    tasks_to_judge = db.get_tasks_for_run(run_key, status_filter='generated')
    if tasks_to_judge:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for task in tasks_to_judge:
                base_prompt = creative_prompts[task.prompt_id]["writing_prompt"]
                task_controller = CreativeWritingTask(task)
                futures.append(executor.submit(
                    task_controller.judge,
                    judge_models,
                    judge_prompt_template,
                    creative_writing_criteria,
                    negative_criteria,
                    base_prompt
                ))

            for future in tqdm(list(futures), desc="Judging creative pieces"):
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"An error occurred during judging future execution: {e}", exc_info=True)
    else:
        logging.info("No tasks require judging.")

    # --- 5. Final Scoring and ELO ---
    compute_benchmark_results_creative(run_key, negative_criteria)

    if run_elo:
        logging.info("Starting ELO analysis...")
        try:
            # ELO function now reads from and writes to the database
            final_elo_snapshot, error_msg = run_elo_analysis_creative(
                run_key=run_key,
                test_model=test_model,
                judge_models=judge_models,
                writing_prompts=creative_prompts,
                concurrency=num_threads
            )
            if error_msg:
                logging.error(f"ELO analysis finished with an error: {error_msg}")

            # Update run with ELO results
            current_run = db.get_run(run_key)
            results_dict = current_run.results or {}
            bench_results = results_dict.get("benchmark_results", {})
            if test_model in final_elo_snapshot:
                bench_results["elo_raw"] = final_elo_snapshot[test_model].get("elo")
                bench_results["elo_normalized"] = final_elo_snapshot[test_model].get("elo_norm")
            else:
                bench_results["elo_raw"] = "Error"
                bench_results["elo_normalized"] = "Error"
            results_dict["benchmark_results"] = bench_results
            db.update_run(run_key, {"results": results_dict})

        except Exception as e:
            logging.error(f"ELO analysis failed critically: {e}", exc_info=True)

    # --- 6. Finalize Run ---
    db.update_run(run_key, {"status": "completed", "end_time": datetime.now(timezone.utc)})
    logging.info(f"Run {run_key} marked as completed.")
    return run_key