#!/usr/bin/env python3
import argparse
import os
import sys
import getpass
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine.url import make_url
from alembic import command
from alembic.config import Config

HERE = Path(__file__).resolve().parent
ALEMBIC_INI = HERE.parent / "alembic.ini"  # project-root/alembic.ini

BANNER = """
=====================================================================
=  !!! DANGER !!!  THIS WILL DROP **ALL OBJECTS** IN SCHEMA PUBLIC  =
=                                                                   =
=  ACTION: RESET DATABASE SCHEMA                                     =
=  TARGET: {driver}://{user}@{host}:{port}/{db}                 =
=====================================================================
"""

def get_alembic_config() -> Config:
    cfg = Config(str(ALEMBIC_INI))
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise SystemExit("DATABASE_URL is not set. Put it in .env or export it.")
    cfg.set_main_option("sqlalchemy.url", db_url)
    cfg.set_main_option("script_location", str(HERE))
    return cfg

def reset_schema(db_url: str):
    eng = create_engine(db_url, isolation_level="AUTOCOMMIT")
    with eng.connect() as con:
        con.execute(text("DROP SCHEMA IF EXISTS public CASCADE;"))
        con.execute(text("CREATE SCHEMA public;"))
        con.execute(text("GRANT ALL ON SCHEMA public TO PUBLIC;"))
        con.execute(text("GRANT ALL ON SCHEMA public TO CURRENT_USER;"))

def safety_checks(url):
    name = (url.database or "").lower()
    if any(tok in name for tok in ("prod", "production", "primary")):
        print(f"Refusing to operate on apparent production database: {url.database}", file=sys.stderr)
        sys.exit(3)

def confirm_reset(url, force: bool):
    if force:
        return
    if not sys.stdin.isatty():
        print("Refusing to reset without TTY confirmation. Use --force for non-interactive.", file=sys.stderr)
        sys.exit(2)

    host = url.host or "(local)"
    port = url.port or "(default)"
    user = url.username or getpass.getuser()
    print(BANNER.format(driver=url.drivername, user=user, host=host, port=port, db=url.database))
    expected = f"RESET {url.database}"
    prompt = f"Type exactly:  {expected}  to proceed, or press Enter to cancel: "
    resp = input(prompt)
    if resp != expected:
        print("Canceled. No changes made.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Ensure DB schema is up to date (autogenerate + upgrade).")
    parser.add_argument("--reset-db", action="store_true",
                        help="Dangerous: drop and recreate the public schema before migrating.")
    parser.add_argument("--force", action="store_true",
                        help="Skip interactive confirmation (CI only).")
    parser.add_argument("--message", "-m", default="autogen",
                        help="Migration message when autogenerating.")
    args = parser.parse_args()

    load_dotenv()
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("DATABASE_URL is not set. Put it in .env or export it.", file=sys.stderr)
        sys.exit(1)

    url = make_url(db_url)
    safety_checks(url)

    if args.reset_db:
        confirm_reset(url, args.force)
        print("Resetting schema 'public' …")
        reset_schema(db_url)

    cfg = get_alembic_config()

    print("Autogenerating migration from utils.db_schema …")
    try:
        command.revision(cfg, message=args.message, autogenerate=True)
    except Exception as e:
        print(f"[info] revision step: {e}")

    print("Upgrading to head …")
    command.upgrade(cfg, "head")
    print("Done.")

if __name__ == "__main__":
    main()
