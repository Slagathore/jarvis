"""
JARVIS — Ambient Home AI
========================
Mission: Entry point for the JARVIS system. Loads environment variables and YAML
         config, configures structured logging via Loguru, instantiates the
         Orchestrator, and launches the async event loop.

         This file deliberately contains almost no logic — that all lives in
         core/orchestrator.py. This file's only job is to boot cleanly.

Modules: main.py
Classes: (none)
Functions:
    main()   — Load config, setup logging, instantiate Orchestrator, run.

Variables:
    CONFIG_PATH  — Path to config.yaml relative to this file
    LOG_DIR      — Directory for log file rotation

#todo: Add --config CLI flag to point at alternate config file
#todo: Add --dry-run flag that loads modules but doesn't start microphone
#todo: Add --log-level CLI override for debugging without editing config.yaml
#todo: Add process lock file so you can't accidentally launch two instances
"""

import asyncio
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from loguru import logger

# ── Constants ─────────────────────────────────────────────────────────────

CONFIG_PATH = Path(__file__).parent / "config.yaml"
LOG_DIR = Path(__file__).parent / "data"


def _configure_event_loop_policy() -> None:
    """Use the selector loop on Windows for libraries that need add_reader/add_writer."""
    if sys.platform == "win32" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def _setup_logging(log_level: str) -> None:
    """
    Configure Loguru with console + rotating file output.
    Removes the default handler and replaces with our own.
    """
    logger.remove()  # Remove default stderr handler

    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
        "<level>{message}</level>"
    )

    # Console (colorized)
    logger.add(
        sys.stderr,
        format=log_format,
        level=log_level,
        colorize=True,
    )

    # Rotating file log
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(LOG_DIR / "jarvis_{time:YYYY-MM-DD}.log"),
        format=log_format,
        level=log_level,
        rotation="00:00",   # New file each midnight
        retention="14 days",
        compression="zip",
        colorize=False,
    )


def _load_config() -> dict:
    """Load and return the YAML config. Raises on missing file or parse error."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {CONFIG_PATH}. "
            "Run python scripts/setup.py to validate your environment."
        )

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not config:
        raise ValueError("config.yaml is empty or invalid.")

    # Overlay any MQTT credentials from .env
    if os.getenv("MQTT_USERNAME"):
        config["mqtt"]["username"] = os.getenv("MQTT_USERNAME")
    if os.getenv("MQTT_PASSWORD"):
        config["mqtt"]["password"] = os.getenv("MQTT_PASSWORD")

    return config


async def main() -> None:
    """
    JARVIS boot sequence:
      1. Load .env
      2. Load config.yaml
      3. Configure Loguru
      4. Create and run the Orchestrator
    """
    # Load .env (silently OK if missing — production envs may inject directly)
    load_dotenv()

    # Load config
    config = _load_config()

    # Setup logging (use level from config, fallback to INFO)
    log_level = config.get("system", {}).get("log_level", "INFO").upper()
    _setup_logging(log_level)

    logger.info("=" * 60)
    logger.info(
        f"  {config['system']['name']} v{config['system']['version']} — Starting"
    )
    logger.info("=" * 60)

    # Import here so Loguru is configured before any module-level loggers fire
    from core.orchestrator import Orchestrator

    orchestrator = Orchestrator(config)

    try:
        await orchestrator.run()
    except KeyboardInterrupt:
        logger.info("[Main] Keyboard interrupt — shutting down.")
    except Exception as e:
        logger.critical(f"[Main] Fatal error: {e}")
        raise


if __name__ == "__main__":
    try:
        _configure_event_loop_policy()
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
