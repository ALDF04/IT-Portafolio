from dotenv import dotenv_values
from pydantic import BaseModel


class Settings(BaseModel):
    BINANCE_API_KEY: str | None = None
    BINANCE_API_SECRET: str | None = None
    TELEGRAM_BOT_TOKEN: str | None = None
    TELEGRAM_CHAT_ID: str | None = None
    REDIS_URL: str | None = None  # ej: redis://localhost:6379/0


def load_settings() -> 'Settings':
    return Settings(**dotenv_values('.env'))
