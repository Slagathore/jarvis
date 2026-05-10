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
import faulthandler
import os
import signal
import sys
from pathlib import Path

# Dump a Python + native traceback to stderr on SIGSEGV / 0xC0000005 etc.
# Costs nothing at runtime; pays for itself the first time something native
# (PyAV/libav, cv2, CUDA, pycaw) crashes the interpreter.
faulthandler.enable()

# ── Native library noise suppression ──────────────────────────────────────
# Both cv2 and PyAV bundle their own copy of libav; both write H.264 decoder
# warnings (`non-existing PPS X referenced`, `decode_slice_header error`,
# `no frame!`, `left block unavailable for requested intra4x4 mode`) directly
# to stderr while a stream is mid-GOP after a cold open or reconnect. Tens
# of lines per second can drown the actual log.
#
# These vars MUST be set before any import that pulls cv2 / PyAV — they
# tap into libav's static log-level config at first use. We use os.environ
# direct assignment (not setdefault) because opencv-python on Windows is
# inconsistent about honoring the string form; the integer form parsed
# through libav's av_log_set_level is the reliable knob.
#
#   OPENCV_FFMPEG_LOGLEVEL=-8   — AV_LOG_QUIET, libav emits NOTHING.
#                                 Numeric form to avoid the string-parsing
#                                 inconsistencies on Windows opencv-python.
#                                 We still get real failures from cv2's
#                                 own Python-side error returns.
#   AV_LOG_FORCE_NOCOLOR=1      — drop ANSI colour codes from any leaked
#                                 stderr so the file log stays readable.
# (PyAV is configured via av.logging.set_level() in wyze_rtsp_mic.py
# right after `import av`, since we don't import av here.)
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ.setdefault("AV_LOG_FORCE_NOCOLOR", "1")

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
    """Load and return the YAML config. Raises on missing file or parse error.

    Performs ${VAR} env-var interpolation (so RTSP URLs like
    rtsp://${WYZE_RTSP_USER}:${WYZE_RTSP_PASSWORD}@... resolve at boot) and
    runs the typed Pydantic validation on the rooms[] block. Validation
    failures raise ConfigError with a pointer to the offending room/field.
    """
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

    # Env-var interpolation + typed-room validation. Imported here (not at
    # top of file) so the config module can use loguru without bootstrapping
    # the logger first.
    from core.config import expand_and_validate
    config, typed_rooms = expand_and_validate(config)
    # Stash the typed rooms under a private key so managers that want
    # autocomplete + Literal exhaustiveness can reach them without re-parsing.
    # Existing modules that read room dicts via config["rooms"] keep working.
    config["_typed_rooms"] = typed_rooms

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


def _install_unix_signal_handlers() -> None:
    """
    On Unix, install SIGTERM as a graceful shutdown that triggers KeyboardInterrupt
    (which the orchestrator's existing run()/finally already handles cleanly).
    On Windows, this is a no-op — Python's default SIGINT handler already raises
    KeyboardInterrupt; installing a custom handler creates duplicate cancellation
    paths that race inside the orchestrator's gather cleanup and recurse.
    """
    if sys.platform == "win32":
        return
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM,):
        try:
            loop.add_signal_handler(
                sig,
                lambda: asyncio.create_task(_raise_kb_interrupt()),
            )
        except (NotImplementedError, RuntimeError) as e:
            logger.debug(f"[Main] Couldn't install handler for {sig}: {e}")


async def _raise_kb_interrupt() -> None:
    """Helper to convert a SIGTERM into the same KeyboardInterrupt path Ctrl+C uses."""
    raise KeyboardInterrupt("SIGTERM")


async def main(args: argparse.Namespace) -> None:
    """
    JARVIS boot sequence:
      1. Load .env
      2. Load config.yaml from --config
      3. Configure Loguru (CLI --log-level overrides config)
      4. If --dry-run, exit before launching the orchestrator
      5. Run the orchestrator (Ctrl+C / SIGTERM trigger graceful shutdown via
         the orchestrator's existing run()/finally cleanup).
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
        from core.orchestrator import Orchestrator  # noqa: F401
        logger.info("[Main] Dry-run successful — all modules importable.")
        return

    from core.orchestrator import Orchestrator
    orchestrator = Orchestrator(config)

    _install_unix_signal_handlers()

    try:
        await orchestrator.run()
    except KeyboardInterrupt:
        logger.info("[Main] Keyboard interrupt — orchestrator shutting down.")
    except Exception as e:
        logger.critical(f"[Main] Fatal error: {e}")
        raise


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
