"""
JARVIS — Ambient Home AI
========================
Mission: Entry point for the JARVIS system. Loads environment variables and YAML
         config, configures structured logging via Loguru, parses CLI flags,
         enforces a single-instance process lock, installs SIGINT/SIGTERM
         shutdown handlers, instantiates the Orchestrator, and launches the
         async event loop.

         This file is the boot harness; all real logic lives in
         core/orchestrator.py.

Modules: main.py
Functions:
    parse_args()                — argparse for --config / --dry-run / --log-level
    main(args)                  — Boot sequence
    _acquire_lock()             — Refuse to start if another live instance exists
    _install_signal_handlers()  — Convert SIGINT/SIGTERM into orchestrator cancel
"""

import argparse
import asyncio
import os
import signal
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from loguru import logger

# ── Constants ─────────────────────────────────────────────────────────────

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"
LOG_DIR             = Path(__file__).parent / "data"
PID_FILE            = Path(__file__).parent / "data" / "jarvis.pid"


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


def _load_config(path: Path) -> dict:
    """Load and return the YAML config. Raises on missing file or parse error."""
    if not path.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {path}. "
            "Run python scripts/setup.py to validate your environment."
        )

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not config:
        raise ValueError(f"{path} is empty or invalid.")

    # Overlay any MQTT credentials from .env
    if os.getenv("MQTT_USERNAME"):
        config["mqtt"]["username"] = os.getenv("MQTT_USERNAME")
    if os.getenv("MQTT_PASSWORD"):
        config["mqtt"]["password"] = os.getenv("MQTT_PASSWORD")

    return config


def parse_args() -> argparse.Namespace:
    """CLI flags. All optional — defaults match historical behavior."""
    parser = argparse.ArgumentParser(
        prog="jarvis",
        description="Ambient home AI — local Python orchestrator.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config.yaml (default: ./config.yaml).",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default=None,
        help="Override system.log_level from config.yaml.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load config + import modules but exit before starting the event loop.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip the process-lock check (use only if you know no other instance is running).",
    )
    return parser.parse_args()


def _process_alive(pid: int) -> bool:
    """Return True if a process with the given PID is currently running."""
    if pid <= 0:
        return False
    try:
        import psutil  # type: ignore[import-not-found]
        return psutil.pid_exists(pid)
    except Exception:
        # Fallback: signal 0 doesn't kill, just probes
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False
        except OSError:
            return False


def _acquire_lock(force: bool) -> None:
    """
    Refuse to start if another Jarvis instance is already running. Two mics
    on the same device + two MQTT clients with the same client ID + two
    Whisper models loaded into the same GPU = mayhem.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if PID_FILE.exists() and not force:
        try:
            existing_pid = int(PID_FILE.read_text().strip() or 0)
        except ValueError:
            existing_pid = 0
        if existing_pid > 0 and existing_pid != os.getpid() and _process_alive(existing_pid):
            print(
                f"[Main] Another Jarvis instance is already running (pid={existing_pid}).\n"
                f"       If you're sure it isn't, delete {PID_FILE} or use --force.",
                file=sys.stderr,
            )
            sys.exit(2)
        # Stale lock — overwrite it
    PID_FILE.write_text(str(os.getpid()))


def _install_signal_handlers(stop_event: asyncio.Event) -> None:
    """
    Convert SIGINT (Ctrl+C) and SIGTERM into a graceful shutdown by setting
    a stop event the orchestrator's run() respects via task cancellation.
    On Windows, only SIGINT is reliably delivered; SIGTERM is best-effort.
    """
    loop = asyncio.get_running_loop()

    def _handler(signum):
        logger.info(f"[Main] Received signal {signum} — initiating graceful shutdown")
        stop_event.set()

    if sys.platform == "win32":
        # Windows asyncio doesn't support add_signal_handler; rely on KeyboardInterrupt
        # for SIGINT and the WinAPI signal module for SIGTERM (best-effort).
        try:
            signal.signal(signal.SIGINT, lambda s, f: _handler(s))
            signal.signal(signal.SIGTERM, lambda s, f: _handler(s))
        except Exception as e:
            logger.debug(f"[Main] Couldn't install Windows signal handlers: {e}")
    else:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _handler, sig)
            except (NotImplementedError, RuntimeError) as e:
                logger.debug(f"[Main] Couldn't install handler for {sig}: {e}")


async def main(args: argparse.Namespace) -> None:
    """
    JARVIS boot sequence:
      1. Load .env
      2. Load config.yaml from --config
      3. Configure Loguru (CLI --log-level overrides config)
      4. If --dry-run, exit before launching the orchestrator
      5. Install SIGINT/SIGTERM graceful shutdown handlers
      6. Run the orchestrator until shutdown signal
    """
    load_dotenv()
    config = _load_config(args.config)

    log_level = (
        args.log_level
        or config.get("system", {}).get("log_level", "INFO")
    ).upper()
    _setup_logging(log_level)

    logger.info("=" * 60)
    logger.info(
        f"  {config['system']['name']} v{config['system']['version']} — Starting"
    )
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("[Main] --dry-run: importing modules then exiting.")
        # Import every major module so missing deps surface here
        from core.orchestrator import Orchestrator  # noqa: F401
        logger.info("[Main] Dry-run successful — all modules importable.")
        return

    from core.orchestrator import Orchestrator
    orchestrator = Orchestrator(config)

    stop_event = asyncio.Event()
    _install_signal_handlers(stop_event)

    run_task = asyncio.create_task(orchestrator.run(), name="orchestrator-run")
    stop_task = asyncio.create_task(stop_event.wait(), name="shutdown-wait")

    try:
        # Whichever finishes first wins — a stop signal cancels the orchestrator.
        await asyncio.wait({run_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
    except KeyboardInterrupt:
        logger.info("[Main] Keyboard interrupt — shutting down.")
        stop_event.set()
    except Exception as e:
        logger.critical(f"[Main] Fatal error: {e}")
        raise
    finally:
        if not run_task.done():
            run_task.cancel()
            try:
                await run_task
            except (asyncio.CancelledError, KeyboardInterrupt):
                pass
            except Exception as e:
                logger.warning(f"[Main] Orchestrator shutdown error: {e}")
        stop_task.cancel()


if __name__ == "__main__":
    _configure_event_loop_policy()
    cli_args = parse_args()
    _acquire_lock(force=cli_args.force)

    try:
        asyncio.run(main(cli_args))
    except KeyboardInterrupt:
        pass
    finally:
        PID_FILE.unlink(missing_ok=True)
