# Database schema / migrations

AquaGuard uses PostgreSQL and SQLAlchemy ORM models in `backend/database/models.py`. For development, the FastAPI startup hook creates missing tables with `Base.metadata.create_all`; it does not select or fall back to SQLite.

For production schema changes, generate and review an Alembic migration before deploying. Do not edit existing tables destructively or store model `.pkl` files in PostgreSQL. Database credentials belong in `DATABASE_URL`, not in migrations.
