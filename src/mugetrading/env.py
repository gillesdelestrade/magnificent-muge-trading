"""API keys live in the local .env file only (never in the database nor in git)."""
from __future__ import annotations

from dotenv import dotenv_values, set_key

from .paths import ENV_PATH


def get_api_key(name: str) -> str:
    return (dotenv_values(ENV_PATH).get(name) or "") if ENV_PATH.exists() else ""


def set_api_key(name: str, value: str) -> None:
    ENV_PATH.touch(exist_ok=True)
    set_key(str(ENV_PATH), name, value)
