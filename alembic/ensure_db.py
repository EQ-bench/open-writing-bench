#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from alembic import command
from alembic.config import Config

HERE = Path(__file__).resolve().parent
ALEMBIC_INI = HERE.parent / "alembic.ini"  # project-root/alembic.ini

def get_alembic_config() -> Config:
    cfg = Config(str(ALEMBIC_INI))
    # Ensure env var flows through even if alembic.ini used a placeholder
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise SystemExit("DATABASE_URL is not set. Put it in .env or export it.")
    cfg.set_main_option("sqlalchemy.url", db_url)
    # typical location is "alembic"
    cfg.set_main_option("script_location", str(HERE))
    return cfg

def reset_schema(db_url: str):
    eng = create_engine(db_url, isolation_level="AUTOCOMMIT")
    with eng.connect() as con:
        # Postgres: drop and recreate the public schema
        con.execute(text("DROP SCHEMA IF EXISTS public CASCADE;"))
        con.execute(text("CREATE SCHEMA public;"))
        # make sure extensions can be created later if needed
        con.execute(text("GRANT ALL ON SCHEMA public TO PUBLIC;"))
        con.execute(text("GRANT ALL ON SCHEMA public TO CURRENT_USER;"))

def main():
    parser = argparse.ArgumentParser(description="Ensure DB schema is up to date (autogenerate + upgrade).")
    parser.add_argument("--reset-db", action="store_true",
                        help="Dangerous: drop and recreate the public schema before migrating.")
    parser.add_argument("--message", "-m", default="autogen",
                        help="Migration message when autogenerating.")
    args = parser.parse_args()

    load_dotenv()
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("DATABASE_URL is not set. Put it in .env or export it.", file=sys.stderr)
        sys.exit(1)

    if args.reset_db:
        print("Resetting schema 'public' …")
        reset_schema(db_url)

    cfg = get_alembic_config()

    # 1) autogenerate a migration (no-op if no changes detected)
    print("Autogenerating migration from utils.db_schema …")
    try:
        command.revision(cfg, message=args.message, autogenerate=True)
    except Exception as e:
        # Alembic prints “No changes detected” and returns; exceptions are rare.
        # Keep going to upgrade head regardless.
        print(f"[info] revision step: {e}")

    # 2) upgrade to head
    print("Upgrading to head …")
    command.upgrade(cfg, "head")
    print("Done.")

if __name__ == "__main__":
    main()
