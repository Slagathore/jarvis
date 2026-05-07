"""
JARVIS — Ambient Home AI
========================
Mission: Smoke-test one room's video, mic, and speaker end-to-end. Run after
         flashing a new Wyze cam (or after touching the camera/audio code)
         to confirm all three channels work before declaring the cam "live."

         Steps per room:
           1. Capture one video frame → save to data/test_<room>_frame.jpg.
           2. Stream the mic for ~3 seconds → save to data/test_<room>_audio.wav.
           3. Generate a 440 Hz tone, send to the speaker.
           4. Print PASS / FAIL per stage; exit non-zero on any FAIL.

         The script doesn't run the full orchestrator — it loads config.yaml,
         instantiates only the managers it needs, exercises one room, then
         tears them down. That keeps it fast (~5s) and easy to debug when
         a stage fails (you don't have ten background tasks racing for the
         logger).

Modules: scripts/test_wyze.py
Functions:
    main()                   — Argparse + orchestration of the four stages.
    test_video(cm, room)     — Stage 1
    test_mic(mm, room)       — Stage 2
    test_speaker(sm, room)   — Stage 3

Usage:
    python scripts/test_wyze.py --room bedroom
    python scripts/test_wyze.py --room kitchen --skip-speaker

#todo: Add a --json flag that emits machine-readable results for CI gating
       once we have a CI pipeline that can speak to the real cams (would
       need a fixture cam in the lab).
#todo: Add a stage that exercises end-to-end latency: speak the test tone
       and measure how long until it shows up in the mic capture. Useful
       for tuning the wyze drain count.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import sys
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from dotenv import load_dotenv

# Make the project importable regardless of where the user runs this from.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from core.config import expand_and_validate  # noqa: E402
from modules.vision.camera_manager import CameraManager  # noqa: E402
from modules.voice.mic_manager import MicManager  # noqa: E402
from modules.voice.speaker_manager import SpeakerManager  # noqa: E402

DATA_DIR = _REPO_ROOT / "data"
# Wyze's RTSP server takes a few seconds to release the audio stream after
# the previous client (cv2's video capture) tears down. Without the gap, a
# fresh PyAV connect lands while the cam is still in TEARDOWN and sees zero
# audio packets.
POST_VIDEO_RELEASE_WAIT_S = 5.0
# 5-second mic capture: long enough to see real audio arrive on Wyze
# (which takes 1-2s to start sending after the open) without making the
# test feel slow.
MIC_CAPTURE_SECONDS = 5.0


def _load_config() -> dict:
    """Load config.yaml + .env, run env expansion + room validation."""
    load_dotenv(_REPO_ROOT / ".env")
    cfg_path = _REPO_ROOT / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.yaml not found at {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    config, _typed = expand_and_validate(raw)
    return config


# ── Stage 1: video ───────────────────────────────────────────────────────────


async def test_video(cm: CameraManager, room: str) -> bool:
    print(f"[1/3] Video: capturing one frame from '{room}'...")
    frame = await cm.capture_frame_async(room)
    if frame is None:
        print(f"      FAIL — capture_frame_async returned None")
        return False
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / f"test_{room}_frame.jpg"
    try:
        import cv2
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            print(f"      FAIL — JPEG encode failed")
            return False
        out.write_bytes(buf.tobytes())
    except Exception as e:
        print(f"      FAIL — write {out} failed: {e}")
        return False
    print(f"      PASS — {frame.shape[1]}x{frame.shape[0]} frame saved to {out}")
    return True


# ── Stage 2: mic ─────────────────────────────────────────────────────────────


async def test_mic(mm: MicManager, room: str) -> bool:
    print(f"[2/3] Mic: capturing {MIC_CAPTURE_SECONDS:.0f}s from '{room}'...")
    captured: list[bytes] = []
    sample_rate_box: list[int] = [16000]

    async def cb(chunk: bytes, sample_rate: int) -> None:
        captured.append(chunk)
        sample_rate_box[0] = sample_rate

    started = await mm.start_capture(room, cb)
    if not started:
        print(f"      FAIL — start_capture returned False (no mic configured?)")
        return False

    try:
        await asyncio.sleep(MIC_CAPTURE_SECONDS)
    finally:
        await mm.stop_capture(room)

    if not captured:
        print(f"      FAIL — no audio chunks received in {MIC_CAPTURE_SECONDS:.0f}s")
        return False

    pcm = b"".join(captured)
    out = DATA_DIR / f"test_{room}_audio.wav"
    try:
        with wave.open(str(out), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate_box[0])
            w.writeframes(pcm)
    except Exception as e:
        print(f"      FAIL — write {out} failed: {e}")
        return False

    duration = len(pcm) / 2 / sample_rate_box[0]
    print(
        f"      PASS — {len(captured)} chunk(s), {duration:.1f}s @ "
        f"{sample_rate_box[0]}Hz saved to {out}"
    )
    return True


# ── Stage 3: speaker ─────────────────────────────────────────────────────────


def _generate_tone(freq_hz: float, duration_s: float, sample_rate: int) -> bytes:
    """440 Hz sine for `duration_s`, int16 PCM. Quiet (50% amplitude) so the
    Wyze speaker doesn't blow out — its tiny cone clips hard above ~70%.
    """
    n = int(duration_s * sample_rate)
    t = np.arange(n, dtype=np.float32) / sample_rate
    wave_arr = 0.5 * np.sin(2.0 * math.pi * freq_hz * t)
    return (wave_arr * 32767.0).astype(np.int16).tobytes()


async def test_speaker(sm: SpeakerManager, room: str) -> bool:
    print(f"[3/3] Speaker: playing 1s 440Hz tone in '{room}'...")
    sample_rate = 16000
    pcm = _generate_tone(440.0, 1.0, sample_rate)
    ok = await sm.play(room, pcm, sample_rate)
    if not ok:
        print(f"      FAIL — sm.play returned False (driver unreachable?)")
        return False
    print(f"      PASS — tone delivered without exception")
    return True


# ── Main ─────────────────────────────────────────────────────────────────────


async def amain(args: argparse.Namespace) -> int:
    config = _load_config()
    room = args.room

    # Verify the room exists in config so we fail fast with a clear message
    # instead of "no mic configured for room 'bdroom'" four lines later.
    room_ids = [r.get("id") for r in config.get("rooms", [])]
    if room not in room_ids:
        print(f"FAIL: room '{room}' not in config.yaml. Available: {room_ids}")
        return 2

    cm = CameraManager(config)
    mm = MicManager(config)
    sm = SpeakerManager(config)

    results: list[bool] = []
    try:
        await cm.load()
        if not args.skip_video:
            results.append(await test_video(cm, room))
        else:
            print("[1/3] Video: SKIPPED")

        # Release the cam before the mic test. Wyze V2 + wz_mini_hacks
        # serves the audio + video tracks from one combined RTSP stream
        # — opening a SECOND client to the same path while the first is
        # active gets a connected-but-silent stream. PyAV demux finds the
        # audio stream (no error), then receives zero packets. Closing
        # the cv2 capture here releases the cam so PyAV gets a real audio
        # feed. Production needs a real fix (shared PyAV container — see
        # TODO in modules/vision/camera_manager.py); for the smoke test,
        # this proves each channel works in isolation.
        if not args.skip_video and not args.skip_mic:
            print(
                f"      (releasing cam + waiting {POST_VIDEO_RELEASE_WAIT_S:.0f}s "
                "before mic test — see test_wyze.py docstring)"
            )
            await cm.close()
            await asyncio.sleep(POST_VIDEO_RELEASE_WAIT_S)

        if not args.skip_mic:
            results.append(await test_mic(mm, room))
        else:
            print("[2/3] Mic: SKIPPED")
        if not args.skip_speaker:
            results.append(await test_speaker(sm, room))
        else:
            print("[3/3] Speaker: SKIPPED")
    finally:
        await sm.close()
        await mm.close()
        await cm.close()

    if all(results):
        print(f"\nALL CHECKS PASSED for room '{room}'.")
        return 0
    print(f"\nONE OR MORE CHECKS FAILED for room '{room}'.")
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test one room's video / mic / speaker end-to-end."
    )
    parser.add_argument(
        "--room",
        required=True,
        help="Room ID from config.yaml (e.g. bedroom, kitchen).",
    )
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--skip-mic", action="store_true")
    parser.add_argument(
        "--skip-speaker",
        action="store_true",
        help="Useful when testing without disturbing whoever's in the room.",
    )
    args = parser.parse_args()

    # Selector loop on Windows for the same reason main.py uses it — some
    # libraries (sounddevice's PortAudio bridge in particular) need
    # add_reader/add_writer support.
    if sys.platform == "win32" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        rc = asyncio.run(amain(args))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        rc = 130
    sys.exit(rc)


if __name__ == "__main__":
    main()
