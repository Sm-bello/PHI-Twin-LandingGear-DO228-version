"""Shared path and environment-backed configuration for PHI-Twin / AeroTwin."""

import os
from pathlib import Path
from typing import Dict

BASE_DIR = Path(__file__).resolve().parent


def _load_dotenv(env_path: Path) -> None:
    """Populate os.environ from a local .env file without overriding existing vars."""
    if not env_path.is_file():
        return
    with env_path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, value)


_load_dotenv(BASE_DIR / ".env")

MODEL_PATH = str(BASE_DIR / "AeroTwin_V16_Model.h5")
FINETUNED_MODEL_PATH = str(BASE_DIR / "AeroTwin_V17_finetuned.h5")
SCALER_SEQ_PATH = str(BASE_DIR / "scaler_seq.pkl")
SCALER_SCALAR_PATH = str(BASE_DIR / "scaler_scalar.pkl")
DATASET_PATH = str(BASE_DIR / "AEROTWIN_Do228_V16p1_Dataset.csv")
SPLIT_MAT_PATH = str(BASE_DIR / "train_val_split_V16p1.mat")
MAINTENANCE_LOG_PATH = str(BASE_DIR / "AEROTWIN_MaintenanceLog_V16p1.csv")
FIGURES_DIR = str(BASE_DIR / "figures")


def get_db_config(include_port: bool = True) -> Dict[str, object]:
    """Return PostgreSQL connection kwargs loaded from environment / .env."""
    cfg: Dict[str, object] = {
        "dbname": os.environ.get("AEROTWIN_DB_NAME", "project_db"),
        "user": os.environ.get("AEROTWIN_DB_USER", "postgres"),
        "password": os.environ.get("AEROTWIN_DB_PASSWORD", ""),
        "host": os.environ.get("AEROTWIN_DB_HOST", "localhost"),
    }
    if include_port:
        cfg["port"] = int(os.environ.get("AEROTWIN_DB_PORT", "5432"))
    return cfg


def get_fg_config() -> Dict[str, object]:
    """Return FlightGear UDP host and port configuration."""
    return {
        "host": os.environ.get("AEROTWIN_FG_HOST", "127.0.0.1"),
        "listen_port": int(os.environ.get("AEROTWIN_FG_LISTEN_PORT", "15000")),
        "send_port": int(os.environ.get("AEROTWIN_FG_SEND_PORT", "15001")),
    }
