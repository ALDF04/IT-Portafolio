from utils.config import load_settings


def test_load_settings():
    settings = load_settings()
    assert hasattr(settings, "TELEGRAM_BOT_TOKEN")
