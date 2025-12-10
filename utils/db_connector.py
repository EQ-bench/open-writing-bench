# utils/db_connector.py

import os
import logging
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional

from .db_schema import Base, Run, Task, JudgeResult, EloComparison, EloRating

load_dotenv()

class DBConnector:
    _instance = None
    _engine = None
    _Session = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DBConnector, cls).__new__(cls)
            db_url = os.getenv("DATABASE_URL")
            if not db_url:
                raise ValueError("DATABASE_URL environment variable not set.")
            
            cls._engine = create_engine(db_url)
            cls._Session = sessionmaker(bind=cls._engine, expire_on_commit=False)

            logging.info("Database connector initialized.")
        return cls._instance

    @contextmanager
    def get_session(self) -> Session:
        session = self._Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def init_db(self):
        """Creates all tables in the database. Should only be called once."""
        Base.metadata.create_all(self._engine)
        logging.info("Database tables created (if they didn't exist).")

    # --- Run Management ---
    def get_or_create_run(self, run_key: str, test_model: str, run_config: Dict[str, Any]) -> Run:
        with self.get_session() as session:
            run = session.query(Run).filter_by(run_key=run_key).first()
            if not run:
                logging.info(f"Creating new run: {run_key}")
                run = Run(run_key=run_key, test_model=test_model, run_config=run_config, status='running')
                session.add(run)
            else:
                logging.info(f"Resuming run: {run_key}")
        return run

    def get_run(self, run_key: str) -> Optional[Run]:
        with self.get_session() as session:
            return session.query(Run).filter_by(run_key=run_key).first()

    def get_runs_by_model(self, model_name: str) -> List[Run]:
        """Get all runs for a specific model, ordered by start time (newest first)."""
        with self.get_session() as session:
            return session.query(Run).filter_by(test_model=model_name).order_by(Run.start_time.desc()).all()

    def get_lexical_stats_for_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get the most recent lexical analysis stats for a model.

        Searches through completed runs for this model and returns the
        lexical_analysis from the most recent run that has it.
        """
        with self.get_session() as session:
            runs = session.query(Run).filter_by(test_model=model_name).order_by(Run.start_time.desc()).all()
            for run in runs:
                if run.results and "lexical_analysis" in run.results:
                    return run.results["lexical_analysis"]
        return None

    def update_run(self, run_key: str, updates: Dict[str, Any]):
        with self.get_session() as session:
            session.query(Run).filter_by(run_key=run_key).update(updates)

    # --- Task Management ---
    def get_tasks_for_run(self, run_key: str, status_filter: Optional[str] = None) -> List[Task]:
        with self.get_session() as session:
            query = session.query(Task).filter_by(run_key=run_key)
            if status_filter:
                query = query.filter_by(status=status_filter)
            return query.all()

    def bulk_insert_tasks(self, tasks: List[Task]):
        with self.get_session() as session:
            session.bulk_save_objects(tasks)

    def update_task(self, task_id: int, updates: Dict[str, Any]):
        with self.get_session() as session:
            session.query(Task).filter_by(id=task_id).update(updates)

    def bulk_insert_judge_results(self, results: List[JudgeResult]):
        with self.get_session() as session:
            session.bulk_save_objects(results)

    def reset_judging_for_task(self, task_id: int):
        """Deletes all judge results for a task and resets its status to 'generated'."""
        with self.get_session() as session:
            session.query(JudgeResult).filter_by(task_id=task_id).delete()
            session.query(Task).filter_by(id=task_id).update({"status": "generated", "aggregated_scores": None})

    def reset_all_judging_for_run(self, run_key: str):
        """Resets all judging data for a run, preparing it to re-run from the judging stage.

        This deletes:
        - All JudgeResult entries for tasks in this run
        - All EloComparison entries for this run
        - EloRating entry for the test model
        - Final scores from run results (benchmark_results)

        And resets:
        - Task statuses from 'judged'/'completed' back to 'generated'
        - Task aggregated_scores to None
        """
        with self.get_session() as session:
            # Get the run to find the test model
            run = session.query(Run).filter_by(run_key=run_key).first()
            if not run:
                logging.warning(f"Run {run_key} not found, nothing to reset")
                return

            test_model = run.test_model

            # Get all task IDs for this run
            task_ids = [t.id for t in session.query(Task).filter_by(run_key=run_key).all()]

            if task_ids:
                # Delete all judge results for these tasks
                deleted_judge_results = session.query(JudgeResult).filter(
                    JudgeResult.task_id.in_(task_ids)
                ).delete(synchronize_session='fetch')
                logging.info(f"Deleted {deleted_judge_results} judge results for run {run_key}")

                # Reset task statuses back to 'generated' (only for tasks that had generation completed)
                updated_tasks = session.query(Task).filter(
                    Task.run_key == run_key,
                    Task.status.in_(['judged', 'completed'])
                ).update({"status": "generated", "aggregated_scores": None}, synchronize_session='fetch')
                logging.info(f"Reset {updated_tasks} tasks from 'judged'/'completed' to 'generated' status")

                # Also reset 'error' tasks that have model_response/model_responses back to 'generated'
                # (these may have failed during judging/aggregation, not generation)
                from sqlalchemy import or_
                updated_error_tasks = session.query(Task).filter(
                    Task.run_key == run_key,
                    Task.status == 'error',
                    or_(Task.model_response.isnot(None), Task.model_responses.isnot(None))
                ).update({"status": "generated", "aggregated_scores": None, "error_message": None}, synchronize_session='fetch')
                if updated_error_tasks:
                    logging.info(f"Reset {updated_error_tasks} error tasks (with generation data) to 'generated' status")

            # Delete ELO comparisons for this run
            deleted_elo = session.query(EloComparison).filter_by(run_key=run_key).delete()
            logging.info(f"Deleted {deleted_elo} ELO comparisons for run {run_key}")

            # Delete ELO rating for the test model
            deleted_rating = session.query(EloRating).filter_by(model_name=test_model).delete()
            if deleted_rating:
                logging.info(f"Deleted ELO rating for model {test_model}")

            # Clear benchmark results from run (keep lexical_analysis)
            if run.results:
                results = dict(run.results)
                if "benchmark_results" in results:
                    del results["benchmark_results"]
                    session.query(Run).filter_by(run_key=run_key).update({"results": results})
                    logging.info(f"Cleared benchmark_results from run {run_key}")

    # --- ELO Management ---
    def get_all_elo_comparisons(self) -> List[EloComparison]:
        with self.get_session() as session:
            return session.query(EloComparison).all()

    def get_elo_comparisons_for_models(self, model_names: List[str]) -> List[EloComparison]:
        """Get ELO comparisons involving specific models."""
        with self.get_session() as session:
            return session.query(EloComparison).filter(
                (EloComparison.model_a.in_(model_names)) | 
                (EloComparison.model_b.in_(model_names))
            ).all()

    def insert_elo_comparison(self, comparison_data: Dict[str, Any]):
        """Insert a single ELO comparison."""
        with self.get_session() as session:
            comp = EloComparison(**comparison_data)
            session.add(comp)

    def bulk_insert_elo_comparisons(self, comparisons: List[EloComparison], batch_size: int = 100):
        """Bulk insert ELO comparisons in batches to avoid query size limits."""
        with self.get_session() as session:
            for i in range(0, len(comparisons), batch_size):
                batch = comparisons[i:i + batch_size]
                session.bulk_save_objects(batch)
                session.flush()

    def get_elo_ratings(self) -> Dict[str, EloRating]:
        with self.get_session() as session:
            ratings = session.query(EloRating).all()
            return {r.model_name: r for r in ratings}

    def upsert_elo_ratings(self, ratings_data: Dict[str, Dict[str, Any]]):
        with self.get_session() as session:
            for model_name, data in ratings_data.items():
                existing = session.query(EloRating).filter_by(model_name=model_name).first()
                if existing:
                    for key, value in data.items():
                        setattr(existing, key, value)
                else:
                    new_rating = EloRating(model_name=model_name, **data)
                    session.add(new_rating)

    def get_task_texts_by_keys(self, task_keys: List[tuple]) -> Dict[tuple, Dict[str, Any]]:
        """Fetch task texts for specific (test_model, iteration_index, prompt_id) tuples.

        Args:
            task_keys: List of (test_model, iteration_index, prompt_id) tuples

        Returns:
            Dict mapping (test_model, iteration_index, prompt_id) -> {
                "model_response": str or None,
                "model_responses": list or None
            }
        """
        if not task_keys:
            return {}

        from sqlalchemy import tuple_

        with self.get_session() as session:
            # Build list of (run.test_model, task.iteration_index, task.prompt_id) conditions
            # Group by test_model first for efficiency
            model_to_keys: Dict[str, List[tuple]] = {}
            for test_model, iter_idx, prompt_id in task_keys:
                if test_model not in model_to_keys:
                    model_to_keys[test_model] = []
                model_to_keys[test_model].append((iter_idx, prompt_id))

            results = {}

            for test_model, keys in model_to_keys.items():
                # Get all runs for this model
                run_keys = [r.run_key for r in session.query(Run.run_key).filter_by(test_model=test_model).all()]

                if not run_keys:
                    continue

                # Build (iteration_index, prompt_id) pairs for this model
                iter_prompt_pairs = [(k[0], k[1]) for k in keys]

                # Query tasks with only the text columns we need
                tasks = (
                    session.query(
                        Task.iteration_index,
                        Task.prompt_id,
                        Task.model_response,
                        Task.model_responses
                    )
                    .filter(
                        Task.run_key.in_(run_keys),
                        Task.status == 'completed',
                        tuple_(Task.iteration_index, Task.prompt_id).in_(iter_prompt_pairs)
                    )
                    .all()
                )

                for task in tasks:
                    key = (test_model, task.iteration_index, task.prompt_id)
                    results[key] = {
                        "model_response": task.model_response,
                        "model_responses": task.model_responses
                    }

            return results

# Singleton instance
db = DBConnector()