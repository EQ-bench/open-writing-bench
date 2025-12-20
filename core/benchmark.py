# core/benchmark.py

"""
Core orchestration logic for the Creative Writing Benchmark.

This module contains the main `run_eq_bench_creative` function which manages
the entire lifecycle of a benchmark run using a database backend. It handles
run initialization, task creation, parallelized generation and judging,
final scoring, and ELO analysis.

Supports both single-turn (legacy) and multi-turn (longform) generation modes.
"""

import uuid
import logging
from datetime import datetime, timezone
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from typing import Dict, List, Optional, Any

from utils.db_connector import db
from utils.db_schema import Run, Task
from utils.api import get_client
from core.conversation import CreativeWritingTask, DEFAULT_NUM_CHAPTERS
from core.progress import RunProgress
from core.scoring import (
    compute_single_benchmark_score_creative,
    bootstrap_benchmark_stability_creative,
    aggregate_ensemble_scores_bulk
)
from core.elo import run_elo_analysis_creative
from core.analysis import analyze_task, aggregate_analyses, format_analysis_summary
import re


def strip_rubric_reasoning(prompt_template: str) -> str:
    """
    Remove the reasoning/analysis section from the rubric judging prompt.
    This modifies the prompt to skip the [Analysis] section and go straight to [Scores].
    """
    # Remove the instruction to write comprehensive analysis
    prompt_template = prompt_template.replace(
        "- You are to write a comprehensive analysis of the piece, then give your scores.\n\n",
        ""
    )
    prompt_template = prompt_template.replace(
        "- You are to write a comprehensive analysis of the piece, then give your scores.",
        ""
    )

    # Replace the output format to remove [Analysis] section
    prompt_template = prompt_template.replace(
        "- Output format is:\n\n[Analysis]\n\nWrite your detailed analysis.\n\n[Scores]",
        "- Output format is:\n\n[Scores]"
    )

    return prompt_template


def strip_elo_reasoning(prompt_template: str) -> str:
    """
    Remove the chain-of-thought reasoning field from the ELO pairwise judging prompt.
    This removes the 'chain_of_thought_reasoning' field from the expected JSON output.
    """
    # Remove the chain_of_thought_reasoning line from the JSON format
    prompt_template = re.sub(
        r'"chain_of_thought_reasoning":\s*"[^"]*",?\n?',
        '',
        prompt_template
    )

    return prompt_template


def compute_benchmark_results_creative(run_key: str, negative_criteria: List[str], ensemble_mode: str = 'vote_avg'):
    """
    Gathers all completed tasks from the DB for the run, aggregates their final
    scores, performs bootstrap analysis, and saves the results to the run record.

    Args:
        run_key: The run key identifier
        negative_criteria: List of criteria names that are negative (lower is better)
        ensemble_mode: One of 'vote_avg', 'vote_maj', or 'split'
    """
    logging.info(f"Aggregating ensemble scores for run {run_key} with mode '{ensemble_mode}'...")
    aggregate_ensemble_scores_bulk(run_key, ensemble_mode=ensemble_mode)


    logging.info(f"Calculating final benchmark results for run {run_key}...")
    # Use lightweight query - only need aggregated_scores for scoring
    completed_tasks = db.get_task_scores_for_run(run_key, status_filter='completed')

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
    test_provider: str,
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
    vllm_params_file: Optional[str],
    backend_config: Optional[Dict[str, Any]] = None,
    multiturn: bool = False,
    num_chapters: int = DEFAULT_NUM_CHAPTERS,
    ensemble_mode: str = 'vote_avg',
    n_prompts: Optional[int] = None,
    disable_rubric_reasoning: bool = False,
    disable_elo_reasoning: bool = False
) -> str:
    """
    Main function to run the creative writing benchmark using the database.

    Args:
        test_model: Name/ID of the model to test
        test_provider: Provider for the test model (e.g., 'openai', 'anthropic')
        judge_models: List of judge model names for ensemble judging
        num_threads: Number of parallel threads for generation/judging
        run_id: Optional run ID (generated if not provided)
        creative_prompts_file: Path to JSON file with writing prompts
        creative_criteria_file: Path to file with judging criteria
        negative_criteria_file: Path to file with negative (inverted) criteria
        judge_prompt_file: Path to judge prompt template
        redo_judging: If True, re-judge already judged tasks
        iterations: Number of iterations per prompt
        run_elo: Whether to run ELO analysis
        vllm_params_file: Deprecated - use backend_config instead
        backend_config: Backend-specific configuration dict (e.g., tensor_parallel_size)
        multiturn: If True, use multi-turn generation (planning + chapters)
        num_chapters: Number of chapters for multi-turn mode (default 4)
        ensemble_mode: Ensemble judging mode - 'vote_avg' (default), 'vote_maj', or 'split'
        disable_rubric_reasoning: If True, remove reasoning/analysis section from rubric judging prompts
        disable_elo_reasoning: If True, remove chain-of-thought reasoning from ELO pairwise prompts

    Returns:
        The run_key for this benchmark run
    """
    # --- 1. Initialize Run and Load Assets ---
    run_key = run_id if run_id else str(uuid.uuid4())

    run_config = {
        "judge_models": judge_models,
        "iterations": iterations,
        "creative_prompts_file": creative_prompts_file,
        "creative_criteria_file": creative_criteria_file,
        "negative_criteria_file": negative_criteria_file,
        "judge_prompt_file": judge_prompt_file,
        "vllm_params_file": vllm_params_file,
        "test_model": test_model,
        "test_provider": test_provider,
        "backend_config": backend_config,
        "multiturn": multiturn,
        "num_chapters": num_chapters if multiturn else None,
        "ensemble_mode": ensemble_mode,
    }

    db.get_or_create_run(run_key, test_model, run_config)

    # If redo_judging is set, reset all judging data for this run
    if redo_judging:
        logging.info(f"Resetting all judging data for run {run_key}...")
        db.reset_all_judging_for_run(run_key)

    # Load criteria and prompts from files (original logic)
    creative_writing_criteria = [line.strip() for line in Path(creative_criteria_file).read_text(encoding='utf-8').splitlines() if line.strip()]
    negative_criteria = [line.strip() for line in Path(negative_criteria_file).read_text(encoding='utf-8').splitlines() if line.strip()]
    judge_prompt_template = Path(judge_prompt_file).read_text(encoding='utf-8')

    # Apply reasoning stripping if requested
    if disable_rubric_reasoning:
        judge_prompt_template = strip_rubric_reasoning(judge_prompt_template)
        logging.info("Rubric reasoning disabled - analysis section removed from judging prompt")

    with open(creative_prompts_file, 'r', encoding='utf-8') as f:
        creative_prompts = json.load(f)

    # Limit prompts if n_prompts is specified
    if n_prompts is not None and n_prompts > 0:
        prompt_keys = list(creative_prompts.keys())[:n_prompts]
        creative_prompts = {k: creative_prompts[k] for k in prompt_keys}
        logging.info(f"Limited to {len(creative_prompts)} prompts (--n-prompts={n_prompts})")

    # --- 2. Prepare Tasks ---
    logging.info("Preparing tasks...")
    # Use lightweight query - only need prompt_id and iteration_index to check existence
    existing_task_keys = db.get_task_keys_for_run(run_key)
    existing_tasks_set = {f"{t['prompt_id']}_{t['iteration_index']}" for t in existing_task_keys}
    tasks_to_create = []

    for prompt_key, prompt_obj in creative_prompts.items():
        for i in range(1, iterations + 1):
            task_key = f"{prompt_key}_{i}"
            if task_key not in existing_tasks_set:
                tasks_to_create.append(Task(
                    run_key=run_key,
                    prompt_id=prompt_key,
                    iteration_index=i,
                    status='initialized'
                ))

    if tasks_to_create:
        logging.info(f"Creating {len(tasks_to_create)} new tasks in the database.")
        db.bulk_insert_tasks(tasks_to_create)

    # --- 3. Generation Phase ---
    logging.info("Starting generation phase...")
    # Include 'generating' status to resume interrupted multi-turn generations
    tasks_to_generate = db.get_tasks_for_run(run_key, status_filter='initialized')
    tasks_to_generate.extend(db.get_tasks_for_run(run_key, status_filter='generating'))

    # Initialize progress tracker
    total_tasks = len(creative_prompts) * iterations
    total_turns = total_tasks * (1 + num_chapters) if multiturn else total_tasks
    progress = RunProgress(
        run_key=run_key,
        total_tasks=total_tasks,
        total_turns=total_turns,
    )
    # Set initial progress (account for already-completed tasks)
    already_generated = db.count_tasks_for_run(run_key, status_filter='generated')
    already_generated += db.count_tasks_for_run(run_key, status_filter='judged')
    already_generated += db.count_tasks_for_run(run_key, status_filter='completed')
    if already_generated > 0:
        progress._completed_tasks = already_generated
        if multiturn:
            progress._completed_turns = already_generated * (1 + num_chapters)
        else:
            progress._completed_turns = already_generated
    progress.flush_generation_to_db()

    if tasks_to_generate:
        # Ensure backend connection pool matches thread concurrency
        effective_backend_config = backend_config.copy() if backend_config else {}
        if "max_concurrent" not in effective_backend_config:
            effective_backend_config["max_concurrent"] = num_threads

        test_model_client = get_client(test_model, client_type='test',
                               vllm_params_file=vllm_params_file,
                               test_provider=test_provider,
                               backend_config=effective_backend_config)

        if multiturn:
            # Multi-turn generation: planning + chapters
            # Cannot use batch mode since each turn depends on previous turns
            logging.info(f"Using multi-turn generation with {num_chapters} chapters...")
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = []
                for task in tasks_to_generate:
                    prompt_obj = creative_prompts[task.prompt_id]
                    prompt = prompt_obj.get("prompt") or prompt_obj.get("writing_prompt")
                    category = prompt_obj.get("category", "")
                    task_controller = CreativeWritingTask(task)
                    futures.append(executor.submit(
                        task_controller.generate_multiturn,
                        test_model_client,
                        prompt,
                        category=category,
                        num_chapters=num_chapters,
                        progress=progress,
                    ))
                # Process futures and periodically flush progress
                completed_count = 0
                for future in tqdm(list(futures), desc="Generating multi-turn pieces"):
                    try:
                        future.result()
                    except Exception as e:
                        logging.error(f"An error occurred during multi-turn generation: {e}", exc_info=True)
                    completed_count += 1
                    # Flush progress every 5 tasks
                    if completed_count % 5 == 0:
                        progress.flush_generation_to_db()
        else:
            # Single-turn generation (legacy mode)
            # if the client supports batch, submit in batches; else keep thread pool
            supports_batch = hasattr(test_model_client, "generate_many")

            if supports_batch:
                # prepare prompts in original order
                prompts = []
                task_ids = []
                for task in tasks_to_generate:
                    prompt_obj = creative_prompts[task.prompt_id]
                    prompt = prompt_obj.get("prompt") or prompt_obj.get("writing_prompt")
                    prompts.append(prompt)
                    task_ids.append(task.id)

                # chunk to avoid giant payloads; reuse args.threads as chunk-size heuristic
                chunk = max(1, min(len(prompts), num_threads))
                for i in tqdm(range(0, len(prompts), chunk), desc="Generating creative pieces"):
                    sub_prompts = prompts[i:i+chunk]
                    sub_task_ids = task_ids[i:i+chunk]
                    try:
                        outputs = test_model_client.generate_many(sub_prompts, temperature=0.7, max_tokens=4000)
                    except Exception as e:
                        logging.error(f"Batch generate failed for slice {i}:{i+chunk}: {e}", exc_info=True)
                        # fall back to per-item using single-generate
                        outputs = []
                        for p in sub_prompts:
                            try:
                                outputs.append(test_model_client.generate(prompt=p, temperature=0.7, max_tokens=4000))
                            except Exception as e2:
                                outputs.append(f"[ERROR] {e2}")

                    # persist results and update progress
                    for tid, text in zip(sub_task_ids, outputs):
                        if isinstance(text, str) and not text.startswith("[ERROR]") and len(text.strip()) >= 500:
                            db.update_task(tid, {"model_response": text.strip(), "status": "generated", "error_message": None})
                            progress.inc_completed_tasks()
                            progress.inc_completed_turns()
                        else:
                            db.update_task(tid, {"status": "error", "error_message": str(text) if isinstance(text, str) else "generation error"})
                            progress.inc_generation_errors()
                    progress.flush_generation_to_db()

            else:
                # legacy threaded path
                with ThreadPoolExecutor(max_workers=num_threads) as executor:
                    futures = []
                    for task in tasks_to_generate:
                        prompt_obj = creative_prompts[task.prompt_id]
                        prompt = prompt_obj.get("prompt") or prompt_obj.get("writing_prompt")
                        task_controller = CreativeWritingTask(task)
                        future = executor.submit(task_controller.generate_creative_piece, test_model_client, prompt)
                        futures.append(future)
                    completed_count = 0
                    for future in tqdm(list(futures), desc="Generating creative pieces"):
                        try:
                            future.result()
                            # Check if task succeeded or failed by querying its status
                            # (generate_creative_piece updates DB directly)
                            progress.inc_completed_tasks()
                            progress.inc_completed_turns()
                        except Exception as e:
                            logging.error(f"An error occurred during generation future execution: {e}", exc_info=True)
                            progress.inc_generation_errors()
                        completed_count += 1
                        if completed_count % 5 == 0:
                            progress.flush_generation_to_db()
    else:
        logging.info("No tasks require generation.")

    # Final flush after generation phase
    progress.flush_generation_to_db()

    # --- 4. Lexical Analysis (before judging so stats are available for judge prompts) ---
    logging.info("Running lexical analysis...")
    generated_tasks_for_analysis = db.get_tasks_for_run(run_key, status_filter='generated')
    # Also include already judged/completed tasks
    generated_tasks_for_analysis.extend(db.get_tasks_for_run(run_key, status_filter='judged'))
    generated_tasks_for_analysis.extend(db.get_tasks_for_run(run_key, status_filter='completed'))

    # Build per-task lexical stats dict for use in judging
    task_lexical_stats: Dict[str, Any] = {}
    if generated_tasks_for_analysis:
        analyses = []
        for task in generated_tasks_for_analysis:
            try:
                analysis = analyze_task(task)
                if analysis:
                    analyses.append(analysis)
                    task_lexical_stats[task.id] = analysis
            except Exception as e:
                logging.warning(f"Lexical analysis failed for task {task.id}: {e}")

        if analyses:
            aggregated = aggregate_analyses(analyses)
            logging.info(f"\n{format_analysis_summary(aggregated)}")

            # Save to run results
            current_run = db.get_run(run_key)
            results_dict = current_run.results or {}
            results_dict["lexical_analysis"] = dict(aggregated)
            db.update_run(run_key, {"results": results_dict})
        else:
            logging.warning("No tasks available for lexical analysis.")
    else:
        logging.info("No generated tasks for lexical analysis.")

    # --- 5. Rubric Judging Phase ---
    logging.info("Starting rubric judging phase...")
    progress.set_phase("rubric_judging")
    rubric_judging_cost = 0.0
    tasks_to_judge = db.get_tasks_for_run(run_key, status_filter='generated')
    progress.set_rubric_total(len(tasks_to_judge))
    progress.flush_judging_to_db()

    if tasks_to_judge:
        # Sort tasks by ID for reproducibility (important for split mode)
        tasks_to_judge = sorted(tasks_to_judge, key=lambda t: t.id)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for idx, task in enumerate(tasks_to_judge):
                prompt_obj = creative_prompts[task.prompt_id]
                base_prompt = prompt_obj.get("prompt") or prompt_obj.get("writing_prompt")
                task_controller = CreativeWritingTask(task)

                # In split mode, assign each task to one judge (round-robin)
                if ensemble_mode == 'split':
                    assigned_judge_idx = idx % len(judge_models)
                    task_judge_models = [judge_models[assigned_judge_idx]]
                else:
                    task_judge_models = judge_models

                # Get precomputed lexical stats for this task
                precomputed_stats = task_lexical_stats.get(task.id)

                futures.append(executor.submit(
                    task_controller.judge,
                    task_judge_models,
                    judge_prompt_template,
                    creative_writing_criteria,
                    negative_criteria,
                    base_prompt,
                    lexical_stats=precomputed_stats,
                ))

            completed_count = 0
            for future in tqdm(list(futures), desc="Judging creative pieces"):
                try:
                    task_cost = future.result()
                    if task_cost:
                        rubric_judging_cost += task_cost
                    progress.inc_rubric_completed()
                except Exception as e:
                    logging.error(f"An error occurred during judging future execution: {e}", exc_info=True)
                    progress.inc_rubric_errors()
                completed_count += 1
                if completed_count % 5 == 0:
                    progress.flush_judging_to_db()

        logging.info(f"Rubric judging complete. Total cost: ${rubric_judging_cost:.4f}")
    else:
        logging.info("No tasks require judging.")

    # Flush rubric judging progress
    progress.flush_judging_to_db()

    # --- 6. Final Scoring and ELO ---
    compute_benchmark_results_creative(run_key, negative_criteria, ensemble_mode=ensemble_mode)

    elo_judging_cost = 0.0
    if run_elo:
        # Check task success rate before running ELO
        completed_count = db.count_tasks_for_run(run_key, status_filter='completed')
        expected_task_count = len(creative_prompts) * iterations
        success_rate = completed_count / expected_task_count if expected_task_count > 0 else 0

        if success_rate < 0.80:
            error_msg = (
                f"ELO analysis skipped: task success rate {success_rate:.1%} "
                f"({completed_count}/{expected_task_count}) is below 80% threshold"
            )
            logging.error(error_msg)
            # Update run with error
            current_run = db.get_run(run_key)
            results_dict = current_run.results or {}
            bench_results = results_dict.get("benchmark_results", {})
            bench_results["elo_raw"] = "Error"
            bench_results["elo_normalized"] = "Error"
            bench_results["elo_error"] = error_msg
            results_dict["benchmark_results"] = bench_results
            db.update_run(run_key, {"results": results_dict})
            raise RuntimeError(error_msg)

        logging.info(f"Starting ELO analysis... (task success rate: {success_rate:.1%})")
        progress.set_phase("elo_judging")
        progress.flush_judging_to_db()
        try:
            # ELO function now reads from and writes to the database
            final_elo_snapshot, error_msg, elo_judging_cost = run_elo_analysis_creative(
                run_key=run_key,
                test_model=test_model,
                judge_models=judge_models,
                writing_prompts=creative_prompts,
                concurrency=num_threads,
                disable_elo_reasoning=disable_elo_reasoning,
                ensemble_mode=ensemble_mode,
                progress=progress,
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

    # --- 7. Store judging costs in results ---
    total_judging_cost = rubric_judging_cost + elo_judging_cost
    current_run = db.get_run(run_key)
    results_dict = current_run.results or {}
    results_dict["judging_costs"] = {
        "rubric_judging_cost_usd": round(rubric_judging_cost, 6),
        "elo_judging_cost_usd": round(elo_judging_cost, 6),
        "total_judging_cost_usd": round(total_judging_cost, 6),
    }
    db.update_run(run_key, {"results": results_dict})
    logging.info(f"Total judging cost: ${total_judging_cost:.4f} (rubric: ${rubric_judging_cost:.4f}, elo: ${elo_judging_cost:.4f})")

    # --- 8. Finalize Run ---
    db.update_run(run_key, {"status": "completed", "end_time": datetime.now(timezone.utc)})
    logging.info(f"Run {run_key} marked as completed.")
    return run_key