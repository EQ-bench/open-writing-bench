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
            cls._Session = sessionmaker(bind=cls._engine)
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

    # --- ELO Management ---
    def get_all_elo_comparisons(self) -> List[EloComparison]:
        with self.get_session() as session:
            return session.query(EloComparison).all()

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

# Singleton instance
db = DBConnector()