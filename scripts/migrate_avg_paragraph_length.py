#!/usr/bin/env python3
"""
Migration script to update avg_paragraph_length in runs.results.lexical_analysis
from sentences-per-paragraph to words-per-paragraph.

This script:
1. Finds all completed runs with lexical_analysis in results
2. For each run, gets all completed tasks and extracts text
3. Recalculates avg_paragraph_length using words instead of sentences
4. Updates the runs.results JSON in the database
"""

import os
import sys
import re
import logging
from copy import deepcopy

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from utils.db_schema import Run, Task

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs (separated by blank lines)."""
    paragraphs = re.split(r'\n\s*\n', text.strip())
    return [p.strip() for p in paragraphs if p.strip()]


def compute_avg_paragraph_length_words(text: str) -> float:
    """Compute average paragraph length in words."""
    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return 0.0

    word_counts = []
    for para in paragraphs:
        words = para.split()
        word_counts.append(len(words))

    return sum(word_counts) / len(word_counts) if word_counts else 0.0


def get_text_from_task(task: Task) -> str:
    """Extract full text from a task (handles both single and multi-turn).

    For multi-turn, only includes chapter turns (not planning turns).
    """
    # Multi-turn: concatenate all chapter assistant responses
    if task.model_responses:
        texts = []
        for turn in task.model_responses:
            # Only include chapter turns, not planning
            if turn.get("turn_type") == "chapter" and turn.get("assistant_response"):
                texts.append(turn["assistant_response"])
        return "\n\n".join(texts)
    # Single-turn
    elif task.model_response:
        return task.model_response
    return ""


def main():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        logger.error("DATABASE_URL environment variable not set")
        sys.exit(1)

    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)

    with Session() as session:
        # Find all runs with lexical_analysis in results
        runs = session.query(Run).filter(
            Run.results.isnot(None),
            Run.status == 'completed'
        ).all()

        logger.info(f"Found {len(runs)} completed runs to check")

        updated_count = 0
        for run in runs:
            if not run.results or "lexical_analysis" not in run.results:
                continue

            # Get all completed tasks for this run
            tasks = session.query(Task).filter(
                Task.run_key == run.run_key,
                Task.status == 'completed'
            ).all()

            if not tasks:
                logger.warning(f"Run {run.run_key}: no completed tasks found, skipping")
                continue

            # Calculate average paragraph length across all tasks
            all_para_lengths = []
            for task in tasks:
                text = get_text_from_task(task)
                if text:
                    paragraphs = _split_paragraphs(text)
                    for para in paragraphs:
                        words = para.split()
                        if words:
                            all_para_lengths.append(len(words))

            if not all_para_lengths:
                logger.warning(f"Run {run.run_key}: no paragraphs found, skipping")
                continue

            new_avg = sum(all_para_lengths) / len(all_para_lengths)
            old_avg = run.results["lexical_analysis"].get("avg_paragraph_length", 0)

            # Update the results
            new_results = deepcopy(run.results)
            new_results["lexical_analysis"]["avg_paragraph_length"] = round(new_avg, 1)

            run.results = new_results
            session.add(run)

            logger.info(f"Run {run.run_key} ({run.test_model}): {old_avg:.1f} -> {new_avg:.1f} words/para")
            updated_count += 1

        session.commit()
        logger.info(f"Updated {updated_count} runs")


if __name__ == "__main__":
    main()
