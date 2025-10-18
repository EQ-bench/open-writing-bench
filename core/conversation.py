# core/conversation.py

"""
Contains the CreativeWritingTask class, which manages the lifecycle of a
single prompt-iteration task: generation and judging.

This class is now a stateless controller that operates on a Task object from the
database, persisting state changes directly to the DB instead of holding them
in memory.
"""

import time
import logging
from typing import Dict, Any, List, Optional

from utils.db_connector import db
from utils.db_schema import Task, JudgeResult, Run
from utils.api import get_client
from core.scoring import parse_judge_scores_creative

class CreativeWritingTask:
    """
    A controller for a single creative writing task. It orchestrates the
    generation of a creative piece and its subsequent judging by an ensemble
    of models. All state is read from and written to the database.
    """

    def __init__(self, db_task: Task):
        """
        Initializes the controller with a SQLAlchemy Task object.
        """
        self.db_task = db_task

    def generate_creative_piece(self, test_model_client, base_prompt: str, seed_modifier: str):
        """
        Generates a creative piece using the provided test model client.
        Retries on short outputs and saves the result or error to the database.
        """
        if self.db_task.status in ["generated", "judged", "completed", "error"]:
            logging.debug(f"Skipping generation for task {self.db_task.id}, status is '{self.db_task.status}'.")
            return

        db.update_task(self.db_task.id, {"status": "generating"})
        final_prompt = base_prompt.replace("<SEED>", seed_modifier)
        
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                response = test_model_client.generate(
                    prompt=final_prompt,
                    temperature=0.7,
                    max_tokens=4000,
                    min_p=0.1
                )
                
                if len(response.strip()) < 500:
                    if attempt < max_attempts:
                        logging.warning(f"Generated text too short ({len(response.strip())} chars) for task {self.db_task.id}, retry {attempt}/{max_attempts}")
                        time.sleep(1)
                        continue
                    else:
                        raise ValueError(f"Generated text too short after {max_attempts} attempts.")
                
                # Success
                updates = {"model_response": response.strip(), "status": "generated", "error_message": None}
                db.update_task(self.db_task.id, updates)
                logging.debug(f"Successfully generated text for task {self.db_task.id}.")
                return

            except Exception as e:
                logging.error(f"Generation error for task {self.db_task.id} on attempt {attempt}/{max_attempts}: {e}", exc_info=True)
                if attempt >= max_attempts:
                    updates = {"status": "error", "error_message": f"Generation failed after {max_attempts} attempts: {str(e)}"}
                    db.update_task(self.db_task.id, updates)
                time.sleep(1)

    def judge(
        self,
        judge_model_names: List[str],
        judge_prompt_template: str,
        creative_writing_criteria: List[str],
        negative_criteria: List[str],
        base_prompt: str
    ):
        """
        Judges the generated piece with an ensemble of models.
        Fetches the model response from the database and saves all individual
        judge results back to the database.
        """
        if self.db_task.status != "generated":
            logging.warning(f"Cannot judge a task with status '{self.db_task.status}' (ID: {self.db_task.id})")
            return

        model_text = self.db_task.model_response
        if not model_text:
            db.update_task(self.db_task.id, {"status": "error", "error_message": "Cannot judge empty generation"})
            return

        db.update_task(self.db_task.id, {"status": "judging"})
        
        judge_results_to_insert = []
        for i, judge_name in enumerate(judge_model_names):
            try:
                judge_client = get_client(judge_name, client_type='judge')

                final_judge_prompt = judge_prompt_template.format(
                    writing_prompt=base_prompt,
                    test_model_response=model_text,
                    creative_writing_criteria="\n".join(["- " + c for c in creative_writing_criteria]),
                    lower_is_better_criteria=", ".join(negative_criteria),
                )

                judge_resp = judge_client.generate(
                    prompt=final_judge_prompt,
                    temperature=0.0,
                    max_tokens=4096
                )
                scores_dict = parse_judge_scores_creative(judge_resp)

                result = JudgeResult(
                    task_id=self.db_task.id,
                    judge_model_name=judge_name,
                    judge_order_index=i,
                    raw_judge_text=judge_resp,
                    judge_scores=scores_dict
                )
                judge_results_to_insert.append(result)

            except Exception as e:
                logging.error(f"Judge '{judge_name}' failed for task {self.db_task.id}: {e}", exc_info=True)
                result = JudgeResult(
                    task_id=self.db_task.id,
                    judge_model_name=judge_name,
                    judge_order_index=i,
                    raw_judge_text=f"[ERROR: {e}]",
                    judge_scores={"error": str(e)}
                )
                judge_results_to_insert.append(result)

        if judge_results_to_insert:
            db.bulk_insert_judge_results(judge_results_to_insert)

        # Mark as judged; a separate aggregation step will mark it 'completed'
        db.update_task(self.db_task.id, {"status": "judged"})
        logging.debug(f"Finished judging for task {self.db_task.id}.")