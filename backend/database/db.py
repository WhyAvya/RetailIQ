import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Load environment variables from backend/.env
load_dotenv()

# Read the PostgreSQL connection string
DATABASE_URL = os.getenv("DATABASE_URL")

# Create the SQLAlchemy base class for database models
Base = declarative_base()

# Database connection objects will be initialized later
engine = None
SessionLocal = None


def init_db():
    global engine, SessionLocal

    if not DATABASE_URL:
        raise ValueError("DATABASE_URL is not configured")

    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

    SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine
    )