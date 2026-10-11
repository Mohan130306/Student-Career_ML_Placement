"""
Database Connection & Engine Configuration
AI-Powered Student Career & Placement Prediction System

Provides database-agnostic connectivity:
- Defaults to local SQLite (sqlite:///data/placement.db) for zero-config lab work.
- Automatically switches to PostgreSQL when DATABASE_URL is set in environment or .env.
"""

from pathlib import Path
import os
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# Load from environment or default to SQLite
DEFAULT_SQLITE_PATH = Path("data/placement.db").resolve()
DEFAULT_SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
DEFAULT_DB_URL = f"sqlite:///{DEFAULT_SQLITE_PATH}"

DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DB_URL)

# Handle SQLite concurrency connection arguments
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """Dependency generator yielding an active database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initializes all database tables defined in SQLAlchemy models."""
    Base.metadata.create_all(bind=engine)
