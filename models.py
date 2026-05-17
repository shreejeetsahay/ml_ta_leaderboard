from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, Boolean, inspect, text
)
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

Base = declarative_base()

class Team(Base):
    __tablename__ = "teams"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    token = Column(String, unique=True)
    computing_id = Column(String, unique=True)
    last_submission = Column(DateTime, nullable=True)
    last_status_check = Column(DateTime, nullable=True)

class Result(Base):
    """
    One row per submission attempt. We keep legacy 'score' for compatibility
    (we store private ROI-MSE there).
    """
    __tablename__ = "results"

    id = Column(Integer, primary_key=True)
    team_id = Column(Integer, index=True)
    attempt = Column(Integer, index=True)

    status = Column(String, default="pending")
    submitted_at = Column(DateTime, default=datetime.utcnow)  # UTC
    model_size = Column(Float)  # MB

    # Model props
    latent_dim = Column(Integer)      # lower is better
    img_size = Column(Integer)        # should be 256
    grayscale = Column(Boolean)       # False

    # Public (dayTrain)
    public_full_mse = Column(Float)
    public_roi_mse = Column(Float)
    public_roi_n = Column(Integer)

    # Private (daySequence1+2)
    private_full_mse = Column(Float)
    private_roi_mse = Column(Float)
    private_roi_n = Column(Integer)

    # Back-compat
    score = Column(Float)             # mirror of private_roi_mse
    artifact = Column(String)         # filename on disk (optional)

engine = create_engine("sqlite:///leaderboard.db", future=True)
SessionLocal = sessionmaker(bind=engine)

def ensure_schema():
    Base.metadata.create_all(engine)
    insp = inspect(engine)
    cols = {c["name"] for c in insp.get_columns("results")}
    wanted = {
        "latent_dim": "INTEGER",
        "img_size": "INTEGER",
        "grayscale": "BOOLEAN",
        "public_full_mse": "FLOAT",
        "public_roi_mse": "FLOAT",
        "public_roi_n": "INTEGER",
        "private_full_mse": "FLOAT",
        "private_roi_mse": "FLOAT",
        "private_roi_n": "INTEGER",
        "artifact": "VARCHAR(255)",
    }
    with engine.begin() as conn:
        for name, sqltype in wanted.items():
            if name not in cols:
                conn.execute(text(f'ALTER TABLE results ADD COLUMN "{name}" {sqltype}'))
        conn.execute(text('CREATE INDEX IF NOT EXISTS ix_results_team_attempt ON results (team_id, attempt)'))
        conn.execute(text('CREATE INDEX IF NOT EXISTS ix_results_submitted_at ON results (submitted_at)'))

ensure_schema()
