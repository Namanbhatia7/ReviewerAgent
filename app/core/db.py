from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
from app.core.config import settings

# Create engine once
engine = create_engine(
    settings.sqlalchemy_url,
    echo=settings.SQL_ECHO,
    poolclass=QueuePool,
    pool_size=settings.POOL_SIZE,
    max_overflow=settings.POOL_MAX_OVERFLOW,
    pool_pre_ping=True,
    future=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    future=True,
)

@contextmanager
def db_session():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

# For FastAPI DI
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
