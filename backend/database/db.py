"""
backend/database/db.py

SQLAlchemy engine / session wiring for the Flask app: one session per
request, closed on teardown. Tables are created at startup.
"""

from __future__ import annotations

import logging
from pathlib import Path

from flask import Flask, current_app, g
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database.models import Base

logger = logging.getLogger(__name__)

EXTENSION_KEY = "db_sessionmaker"


def init_db(app: Flask, database_url: str) -> None:
    """Create the engine and tables and register per-request sessions."""

    url = make_url(database_url)
    options: dict = {}

    if url.get_backend_name() == "sqlite":
        # Flask serves requests from several threads.
        options["connect_args"] = {"check_same_thread": False}

        if url.database in (None, "", ":memory:"):
            options["poolclass"] = StaticPool  # one shared in-memory database
        else:
            Path(url.database).parent.mkdir(parents=True, exist_ok=True)
    else:
        # Hosted Postgres (Neon / Supabase) closes idle connections.
        options["pool_pre_ping"] = True
        options["pool_recycle"] = 300

    engine = create_engine(url, **options)
    Base.metadata.create_all(engine)
    add_missing_columns(engine)

    app.extensions[EXTENSION_KEY] = sessionmaker(engine, expire_on_commit=False)
    app.teardown_appcontext(_close_session)

    logger.info("Database ready (%s)", url.get_backend_name())


def add_missing_columns(engine: Engine) -> None:
    """
    Add nullable columns that newer code defines but an existing database
    lacks (create_all only creates missing tables). A minimal stand-in for
    migrations: only ever adds nullable columns, never changes or drops.
    """

    inspector = inspect(engine)

    with engine.begin() as connection:
        for table in Base.metadata.sorted_tables:
            if not inspector.has_table(table.name):
                continue

            existing = {column["name"] for column in inspector.get_columns(table.name)}

            for column in table.columns:
                if column.name in existing:
                    continue
                if not column.nullable:
                    raise RuntimeError(
                        f"Column {table.name}.{column.name} is missing and not nullable; "
                        "add it with a real migration."
                    )

                column_type = column.type.compile(dialect=engine.dialect)
                connection.execute(
                    text(f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {column_type}')
                )
                logger.info("Added column %s.%s", table.name, column.name)


def db_session() -> Session:
    """Session for the current request."""

    if "db_session" not in g:
        g.db_session = current_app.extensions[EXTENSION_KEY]()

    return g.db_session


def _close_session(_exc: BaseException | None) -> None:
    session = g.pop("db_session", None)

    if session is not None:
        session.close()
