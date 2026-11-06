import os
from threading import Lock

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Load environment variables without hardcoding credentials.
load_dotenv()

# Read the PostgreSQL connection string
DATABASE_URL = os.getenv("DATABASE_URL")
DATABASE_URL_UNPOOLED = os.getenv("DATABASE_URL_UNPOOLED")

# Create the SQLAlchemy base class for database models
Base = declarative_base()

# Database connection objects will be initialized later
engine = None
SessionLocal = None
_initialization_lock = Lock()


def _psycopg_database_url(database_url: str) -> str:
    if database_url.startswith("postgres://"):
        return database_url.replace("postgres://", "postgresql+psycopg://", 1)
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    if database_url.startswith("postgresql+psycopg://"):
        return database_url
    raise ValueError("DATABASE_URL must use a PostgreSQL connection string")


def init_db():
    global engine, SessionLocal

    if not DATABASE_URL:
        raise ValueError("DATABASE_URL is not configured; PostgreSQL tables were not created")

    engine = create_engine(_psycopg_database_url(DATABASE_URL), pool_pre_ping=True)

    SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
    )

    # Import models only after Base exists so all table metadata is registered.
    from backend.database import models  # noqa: F401

    schema_database_url = DATABASE_URL_UNPOOLED or DATABASE_URL
    schema_engine = create_engine(
        _psycopg_database_url(schema_database_url),
        pool_pre_ping=True,
    )
    try:
        Base.metadata.create_all(bind=schema_engine)
    finally:
        schema_engine.dispose()

    return engine


def get_session_factory():
    """Return the configured session factory, initializing it once if needed."""
    global SessionLocal

    if SessionLocal is None:
        with _initialization_lock:
            if SessionLocal is None:
                init_db()

    return SessionLocal
