"""
backend/database/db.py

SQLAlchemy engine / session wiring for the Flask app: one session per
request, closed on teardown. Tables are created at startup.
"""

from __future__ import annotations

import logging
from pathlib import Path

from flask import Flask, current_app, g
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
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

    app.extensions[EXTENSION_KEY] = sessionmaker(engine, expire_on_commit=False)
    app.teardown_appcontext(_close_session)

    logger.info("Database ready (%s)", url.get_backend_name())


def db_session() -> Session:
    """Session for the current request."""

    if "db_session" not in g:
        g.db_session = current_app.extensions[EXTENSION_KEY]()

    return g.db_session


def _close_session(_exc: BaseException | None) -> None:
    session = g.pop("db_session", None)

    if session is not None:
        session.close()
