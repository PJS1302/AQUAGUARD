from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
MODELS_DIR = Path(os.getenv("AQUAGUARD_MODEL_DIR", str(BACKEND_DIR / "models"))).resolve()
UPLOAD_DIR = Path(os.getenv("AQUAGUARD_UPLOAD_DIR", str(BACKEND_DIR / "data" / "uploads"))).resolve()
PROCESSED_DIR = BACKEND_DIR / "data" / "processed"
EXPERIMENT_DIR = MODELS_DIR / "experiments"
MAX_UPLOAD_MB = int(os.getenv("AQUAGUARD_MAX_UPLOAD_MB", "25"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_ROWS = 500_000
MAX_COLUMNS = 300

for directory in (MODELS_DIR, UPLOAD_DIR, PROCESSED_DIR, EXPERIMENT_DIR):
    directory.mkdir(parents=True, exist_ok=True)
