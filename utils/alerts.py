from __future__ import annotations

import logging
from typing import Any

import requests

from .config import load_settings

logger = logging.getLogger(__name__)


def send_alert(message: str, **kwargs: Any) -> bool:
    """Send a Telegram alert if credentials are configured."""
    settings = load_settings()
    token = settings.TELEGRAM_BOT_TOKEN
    chat_id = settings.TELEGRAM_CHAT_ID
    if not token or not chat_id:
        logger.debug("Telegram credentials not configured; skipping alert for message=%s", message)
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, **kwargs}
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - network failure handling
        logger.error("Failed to send Telegram alert: %s", exc)
        return False
    return True
