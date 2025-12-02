# core/conversation.py

"""
Contains the CreativeWritingTask class, which manages the lifecycle of a
single prompt-iteration task: generation and judging.

This class is now a stateless controller that operates on a Task object from the
database, persisting state changes directly to the DB instead of holding them
in memory.

Supports both single-turn (legacy) and multi-turn (longform) generation modes.
"""

import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from utils.db_connector import db
from utils.db_schema import Task, JudgeResult, Run
from utils.api import get_client
from utils.truncation import truncate_text
from core.scoring import parse_judge_scores_creative

# Multi-turn configuration
DEFAULT_NUM_CHAPTERS = 4


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

    # =========================================================================
    # SINGLE-TURN GENERATION (Legacy)
    # =========================================================================

    def generate_creative_piece(self, test_model_client, prompt: str):
        """
        Generates a creative piece using the provided test model client.
        Retries on short outputs and saves the result or error to the database.
        """
        if self.db_task.status in ["generated", "judged", "completed", "error"]:
            logging.debug(f"Skipping generation for task {self.db_task.id}, status is '{self.db_task.status}'.")
            return

        db.update_task(self.db_task.id, {"status": "generating"})

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                response = test_model_client.generate(
                    prompt=prompt,
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

    # =========================================================================
    # MULTI-TURN GENERATION (Longform)
    # =========================================================================

    def initialize_multiturn_structure(
        self,
        prompt: str,
        num_chapters: int = DEFAULT_NUM_CHAPTERS,
        planning_prompt_template: str = "",
        chapter_first_template: str = "",
        chapter_intermediate_template: str = "",
        chapter_last_template: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Initialize the model_responses structure for multi-turn generation.

        Creates a list of turn dictionaries with:
        - Turn 0: Planning (brainstorm + chapter plans + character outlines)
        - Turns 1-N: Chapters

        Args:
            prompt: The writing prompt
            num_chapters: Number of chapters to generate
            planning_prompt_template: Template for the planning turn
            chapter_first_template: Template for the first chapter
            chapter_intermediate_template: Template for intermediate chapters
            chapter_last_template: Template for the last chapter

        Returns:
            List of turn dictionaries
        """

        turns = []

        # Turn 0: Planning
        planning_user_prompt = planning_prompt_template.replace(
            "{writing_prompt}", prompt
        ).replace("{n_chapters}", str(num_chapters))

        turns.append({
            "turn_type": "planning",
            "turn_index": 0,
            "user_prompt": planning_user_prompt,
            "assistant_response": None,
            "status": "pending",
            "error": None,
            "chapter_number": None
        })

        # Turns 1-N: Chapters
        for chapter_num in range(1, num_chapters + 1):
            if chapter_num == 1:
                template = chapter_first_template
            elif chapter_num == num_chapters:
                template = chapter_last_template.replace("{chapter_number}", str(chapter_num))
            else:
                template = chapter_intermediate_template.replace("{chapter_number}", str(chapter_num))

            turns.append({
                "turn_type": "chapter",
                "turn_index": chapter_num,
                "user_prompt": template,
                "assistant_response": None,
                "status": "pending",
                "error": None,
                "chapter_number": chapter_num
            })

        return turns

    def generate_multiturn(
        self,
        test_model_client,
        prompt: str,
        num_chapters: int = DEFAULT_NUM_CHAPTERS,
        planning_prompt_template: Optional[str] = None,
        chapter_first_template: Optional[str] = None,
        chapter_intermediate_template: Optional[str] = None,
        chapter_last_template: Optional[str] = None,
        max_retries: int = 3
    ):
        """
        Generate a multi-turn creative piece with planning and chapters.

        Flow:
        1. Planning turn (brainstorm, chapter plans, character outlines)
        2. Chapter turns (1 through num_chapters)

        Each turn builds on the conversation history.
        """
        if self.db_task.status in ["generated", "judged", "completed", "error"]:
            logging.debug(f"Skipping multi-turn generation for task {self.db_task.id}, status is '{self.db_task.status}'.")
            return

        # Load default templates if not provided
        data_dir = Path(__file__).parent.parent / "data"
        if planning_prompt_template is None:
            planning_prompt_template = (data_dir / "multiturn_planning_prompt.txt").read_text(encoding="utf-8")
        if chapter_first_template is None:
            chapter_first_template = (data_dir / "multiturn_chapter_first.txt").read_text(encoding="utf-8")
        if chapter_intermediate_template is None:
            chapter_intermediate_template = (data_dir / "multiturn_chapter_intermediate.txt").read_text(encoding="utf-8")
        if chapter_last_template is None:
            chapter_last_template = (data_dir / "multiturn_chapter_last.txt").read_text(encoding="utf-8")

        db.update_task(self.db_task.id, {"status": "generating"})

        # Initialize or resume model_responses structure
        model_responses = self.db_task.model_responses
        if not model_responses:
            model_responses = self.initialize_multiturn_structure(
                prompt=prompt,
                num_chapters=num_chapters,
                planning_prompt_template=planning_prompt_template,
                chapter_first_template=chapter_first_template,
                chapter_intermediate_template=chapter_intermediate_template,
                chapter_last_template=chapter_last_template
            )
            db.update_task(self.db_task.id, {"model_responses": model_responses})

        # Build message history and generate each turn
        for turn_idx, turn in enumerate(model_responses):
            if turn["status"] == "generated":
                continue  # Already generated, skip

            if turn["status"] == "error":
                logging.warning(f"Turn {turn_idx} previously errored for task {self.db_task.id}, skipping.")
                continue

            # Mark turn as generating
            model_responses[turn_idx]["status"] = "generating"
            db.update_task(self.db_task.id, {"model_responses": model_responses})

            # Build conversation history up to this turn
            messages = self._build_conversation_history(model_responses, turn_idx)

            # Generate with retries
            success = False
            for attempt in range(1, max_retries + 1):
                try:
                    response = test_model_client.generate(
                        messages=messages,
                        temperature=0.7,
                        max_tokens=6000 if turn["turn_type"] == "planning" else 4000,
                        min_p=0.1
                    )

                    # Validate response length
                    min_length = 1000 if turn["turn_type"] == "planning" else 500
                    if len(response.strip()) < min_length:
                        if attempt < max_retries:
                            logging.warning(
                                f"Turn {turn_idx} response too short ({len(response.strip())} chars) "
                                f"for task {self.db_task.id}, retry {attempt}/{max_retries}"
                            )
                            time.sleep(1)
                            continue
                        else:
                            raise ValueError(f"Response too short after {max_retries} attempts.")

                    # Success - update the turn
                    model_responses[turn_idx]["assistant_response"] = response.strip()
                    model_responses[turn_idx]["status"] = "generated"
                    model_responses[turn_idx]["error"] = None
                    db.update_task(self.db_task.id, {"model_responses": model_responses})

                    logging.info(
                        f"Task {self.db_task.id}: Generated turn {turn_idx} "
                        f"({turn['turn_type']}) - {len(response.strip())} chars"
                    )
                    success = True
                    break

                except Exception as e:
                    logging.error(
                        f"Generation error for task {self.db_task.id} turn {turn_idx} "
                        f"attempt {attempt}/{max_retries}: {e}",
                        exc_info=True
                    )
                    if attempt >= max_retries:
                        model_responses[turn_idx]["status"] = "error"
                        model_responses[turn_idx]["error"] = str(e)
                        db.update_task(self.db_task.id, {
                            "model_responses": model_responses,
                            "status": "error",
                            "error_message": f"Failed at turn {turn_idx}: {str(e)}"
                        })
                        return
                    time.sleep(1)

            if not success:
                return  # Exit if generation failed

        # All turns generated successfully
        db.update_task(self.db_task.id, {"status": "generated", "error_message": None})
        logging.info(f"Task {self.db_task.id}: Multi-turn generation completed successfully.")

    def _build_conversation_history(
        self,
        model_responses: List[Dict[str, Any]],
        current_turn_idx: int
    ) -> List[Dict[str, str]]:
        """
        Build the conversation history up to and including the current turn's user prompt.

        Args:
            model_responses: List of turn dictionaries
            current_turn_idx: Index of the current turn being generated

        Returns:
            List of message dictionaries for the API call
        """
        messages = []

        # Add all previous turns' user prompts and assistant responses
        for i in range(current_turn_idx):
            turn = model_responses[i]
            messages.append({"role": "user", "content": turn["user_prompt"]})
            if turn["assistant_response"]:
                messages.append({"role": "assistant", "content": turn["assistant_response"]})

        # Add current turn's user prompt
        messages.append({"role": "user", "content": model_responses[current_turn_idx]["user_prompt"]})

        return messages

    def get_chapters_text(self) -> List[str]:
        """
        Extract chapter texts from model_responses, excluding planning.

        Returns:
            List of chapter text strings (chapters only, no planning)
        """
        if not self.db_task.model_responses:
            return []

        chapters = []
        for turn in self.db_task.model_responses:
            if turn["turn_type"] == "chapter" and turn["assistant_response"]:
                chapters.append(turn["assistant_response"])

        return chapters

    def get_planning_text(self) -> Optional[str]:
        """
        Extract the planning text from model_responses.

        Returns:
            Planning text or None if not available
        """
        if not self.db_task.model_responses:
            return None

        for turn in self.db_task.model_responses:
            if turn["turn_type"] == "planning" and turn["assistant_response"]:
                return turn["assistant_response"]

        return None

    def get_full_story_text(self) -> str:
        """
        Get the full story text (all chapters combined, no planning).

        Returns:
            Combined chapter text with chapter markers
        """
        chapters = self.get_chapters_text()
        if not chapters:
            return ""

        parts = []
        for i, chapter in enumerate(chapters, 1):
            parts.append(f"# Chapter {i}\n\n{chapter}")

        return "\n\n---\n\n".join(parts)

    # =========================================================================
    # JUDGING (supports both single-turn and multi-turn)
    # =========================================================================

    def judge(
        self,
        judge_model_names: List[str],
        judge_prompt_template: str,
        creative_writing_criteria: List[str],
        negative_criteria: List[str],
        base_prompt: str,
        max_chars_for_judging: int = 8000,
        truncation_mode: str = "middle"
    ):
        """
        Judges the generated piece with an ensemble of models.

        For multi-turn tasks:
        - Only judges the chapters (excludes planning)
        - Uses middle truncation by default

        For single-turn tasks:
        - Uses end truncation by default

        Fetches the model response from the database and saves all individual
        judge results back to the database.
        """
        if self.db_task.status != "generated":
            logging.warning(f"Cannot judge a task with status '{self.db_task.status}' (ID: {self.db_task.id})")
            return

        # Determine if this is a multi-turn or single-turn task
        is_multiturn = self.db_task.model_responses is not None and len(self.db_task.model_responses) > 0

        if is_multiturn:
            # Multi-turn: get only chapters (exclude planning)
            model_text = self.get_full_story_text()
            truncation_mode = "middle"
        else:
            # Single-turn: use legacy model_response
            model_text = self.db_task.model_response
            truncation_mode = "end"

        if not model_text:
            db.update_task(self.db_task.id, {"status": "error", "error_message": "Cannot judge empty generation"})
            return

        # Apply truncation for judging
        model_text_truncated = truncate_text(model_text, max_chars_for_judging, mode=truncation_mode)

        db.update_task(self.db_task.id, {"status": "judging"})

        judge_results_to_insert = []
        for i, judge_name in enumerate(judge_model_names):
            try:
                judge_client = get_client(judge_name, client_type='judge')

                final_judge_prompt = judge_prompt_template.format(
                    writing_prompt=base_prompt,
                    test_model_response=model_text_truncated,
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
