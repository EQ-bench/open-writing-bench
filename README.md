# Setup db:

```
# ensure .env has DATABASE_URL
# DATABASE_URL=postgresql+psycopg2://USER:PASS@HOST:5432/DBNAME

python alembic/ensure_db.py
```

# Full db reset:

`python alembic/ensure_db.py --reset-db`