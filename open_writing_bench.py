# creative_writing_bench.py

"""
Main entry for the Creative Writing Benchmark with iteration-based generation.
"""
import argparse
import sys
import signal
import logging
import os
from datetime import datetime, timezone
from dotenv import load_dotenv

# Load .env early, before any HuggingFace imports
load_dotenv()

# Set up HuggingFace authentication if HF_TOKEN is available
# This must happen before importing transformers/vllm
_hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if _hf_token:
    # huggingface_hub respects HF_TOKEN env var automatically,
    # but we can also do programmatic login for extra safety
    try:
        from huggingface_hub import login
        login(token=_hf_token, add_to_git_credential=False)
    except ImportError:
        pass  # huggingface_hub not installed, rely on env var
    except Exception:
        pass  # Login failed, continue anyway - env var should work

from utils.logging_setup import setup_logging, get_verbosity
from utils.db_connector import db
from utils.pretty_print import print_postrun_displays

def signal_handler(signum, frame):
    print(f"\n[DEBUG] Signal {signum} caught! Stopping gracefully.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run Creative Writing Benchmark (with iterations).")
    parser.add_argument("--test-model", required=True, help="The model name or identifier for the test model.")
    parser.add_argument("--test-provider", required=True,
                        choices=["http", "openai", "vllm", "vllm_server", "vllm_local", "llamacpp", "llamacpp_local",
                                 "llama_cpp_python", "llama_cpp_server", "transformers", "hf"],
                        help="Backend for the test model. 'http'/'openai' use TEST_API_KEY/TEST_API_URL. "
                             "'vllm' (managed server) / 'vllm_local' (in-process) / 'llama_cpp_server' / 'transformers' run locally.")

    parser.add_argument("--judge-models", required=True, help="Comma-delimited list of judge model names (supports duplicates for stacking).")
    parser.add_argument("--run-id", help="Optional: Resume or create a run with this ID")
    parser.add_argument("--threads", type=int, default=4, help="Number of parallel threads.")
    parser.add_argument("--verbosity", choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'], default="INFO")
    parser.add_argument("--redo-judging", action="store_true", default=False, help="Re-run the judge step on existing items.")
    parser.add_argument("--creative-prompts-file", default="data/open-writing-bench-prompts.json")
    parser.add_argument("--criteria-file", default="data/creative_writing_criteria.txt")
    parser.add_argument("--negative-criteria-file", default="data/negative_criteria.txt")
    parser.add_argument("--judge-prompt-file", default="data/creative_writing_judging_prompt.txt")
    parser.add_argument("--save-interval", type=int, default=2, help="How often to save partial progress.")
    parser.add_argument("--iterations", type=int, default=1, help="How many iteration passes to run (one seed per iteration).")
    parser.add_argument("--vllm-params-file", help="Deprecated: Use --backend-config instead.")
    parser.add_argument("--no-elo", action="store_true", default=False, help="Disable the ELO analysis step.")
    parser.add_argument("--backend-config", type=str, default=None,
                        help="JSON string or path to JSON file with backend-specific configuration. "
                             "Example: '{\"tensor_parallel_size\": 2}' or 'config/backend.json'")
    parser.add_argument("--ensemble-mode", type=str, default="vote_avg",
                        choices=["vote_avg", "vote_maj", "split"],
                        help="Ensemble judging mode: 'vote_avg' averages scores across judges (default), "
                             "'vote_maj' uses majority voting per metric, "
                             "'split' distributes items across judges (no ensemble, depth 1).")
    parser.add_argument("--n-prompts", type=int, default=None,
                        help="Limit the number of prompts to use from the creative prompts file.")

    args = parser.parse_args()
    os.environ["INSPECT_MAX_CONNECTIONS"] = str(args.threads)

    # Parse backend config (JSON string or file path)
    backend_config = None
    if args.backend_config:
        import json
        config_str = args.backend_config.strip()
        if config_str.startswith('{'):
            # JSON string
            try:
                backend_config = json.loads(config_str)
            except json.JSONDecodeError as e:
                logging.error(f"Invalid JSON in --backend-config: {e}")
                sys.exit(1)
        else:
            # File path
            try:
                with open(config_str, 'r') as f:
                    backend_config = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logging.error(f"Failed to load backend config from {config_str}: {e}")
                sys.exit(1)

    setup_logging(get_verbosity(args.verbosity))

    # import after logging is configured
    from core.benchmark import run_eq_bench_creative

    # Hook signals
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    run_elo_flag = not args.no_elo # Determine if ELO should run
    
    # Parse comma-delimited judge models list
    judge_models = [j.strip() for j in args.judge_models.split(',') if j.strip()]
    if not judge_models:
        logging.error("No valid judge models provided.")
        sys.exit(1)

    run_key = run_eq_bench_creative(
        test_model=args.test_model,
        test_provider=args.test_provider,
        judge_models=judge_models,
        num_threads=args.threads,
        run_id=args.run_id,
        creative_prompts_file=args.creative_prompts_file,
        creative_criteria_file=args.criteria_file,
        negative_criteria_file=args.negative_criteria_file,
        judge_prompt_file=args.judge_prompt_file,
        redo_judging=args.redo_judging,
        iterations=args.iterations,
        run_elo=run_elo_flag,
        vllm_params_file=args.vllm_params_file,
        backend_config=backend_config,
        multiturn=True,
        num_chapters=3,
        ensemble_mode=args.ensemble_mode,
        n_prompts=args.n_prompts
    )


    

    # Pretty print summary + centered leaderboards (21 rows incl. ours)
    try:
        with db.get_session() as session:
            print_postrun_displays(session, run_key)
    except Exception as e:
        logging.warning(f"Pretty print failed: {e}")

    logging.info(f"Creative writing benchmark completed. Run key: {run_key}")
    print(f"\nCreative writing benchmark completed. Run key: {run_key}")

if __name__ == "__main__":
    main()