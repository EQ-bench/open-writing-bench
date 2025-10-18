# Setup env:
First, install uv. Then:
uv sync

# Setup db:

```
# make sure .env has DATABASE_URL
# DATABASE_URL=postgresql+psycopg2://USER:PASS@HOST:5432/DBNAME

uv ensure-db
```

# Full db reset (warning! will wipe all data):

`python alembic/ensure_db.py --reset-db`

# Run benchmark

uv run bench --test-model z-ai/glm-4.6 --test-provider openai --judge-models kimi-k2-0905,glm-4.6 --threads 96