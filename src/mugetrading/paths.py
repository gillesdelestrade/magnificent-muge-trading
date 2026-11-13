"""Project paths (everything is local)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = ROOT / "assets"
LOGO_PATH = ASSETS_DIR / "logo.png"
FAVICON_PATH = ASSETS_DIR / "favicon.png"
CONFIG_DIR = ROOT / "config"
FEE_PROFILES_PATH = CONFIG_DIR / "fee_profiles.yaml"
UNIVERSES_PATH = CONFIG_DIR / "universes.yaml"
AI_MODELS_PATH = CONFIG_DIR / "ai_models.yaml"
ALIASES_PATH = CONFIG_DIR / "company_aliases.yaml"
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "mugetrading.sqlite"
AI_CACHE_PATH = DATA_DIR / "ai_cache.sqlite"
ENV_PATH = ROOT / ".env"
BACKUP_DIR = ROOT / "sauvegardes"
