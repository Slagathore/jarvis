"""
JARVIS — Ambient Home AI
========================
Mission: Phase 3 test script. Validates the vision pipeline: camera access,
         light detection, posture analysis, object detection, and scene
         description. Uses the default USB webcam (device 0) if available,
         or falls back to a synthetic frame for unit testing.

         Run: python scripts/test_vision.py

Modules: scripts/test_vision.py
Classes: (none)
Functions:
    test_camera_manager()    — Connect to webcam, capture a frame
    test_light_detector()    — Analyze brightness on captured or synthetic frame
    test_posture_analyzer()  — Run MediaPipe pose on a frame
    test_object_detector()   — Run YOLOv8n detection on a frame
    test_scene_analyzer()    — Call the configured Ollama vision model with a frame
    run_tests()              — Run all tests, print summary

#todo: Add test that saves a debug frame to data/test_frame_<timestamp>.jpg
#todo: Add test for MJPEG stream connection (mock HTTP server)
#todo: Add person detection accuracy test against a known test image
"""

import asyncio
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import yaml
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="WARNING", colorize=True)

GREEN = "\033[92m"
RED   = "\033[91m"
CYAN  = "\033[96m"
BOLD  = "\033[1m"
RESET = "\033[0m"


def ok(msg: str):   print(f"  {GREEN}[OK]{RESET} {msg}")
def fail(msg: str): print(f"  {RED}[X]{RESET} {msg}")
def info(msg: str): print(f"  {CYAN}->{RESET} {msg}")


def _load_config() -> dict:
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _make_synthetic_frame() -> np.ndarray:
    """Create a 480×640 RGB test frame with a gradient. No camera needed."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(480):
        frame[i, :, 0] = int(i * 255 / 480)   # Red gradient
        frame[i, :, 2] = int((480 - i) * 255 / 480)  # Blue inverse gradient
    return frame


async def test_camera_manager() -> tuple[bool, np.ndarray]:
    """Returns (success, frame) — frame may be synthetic if no camera."""
    print(f"\n{BOLD}[1] Camera Manager{RESET}")
    try:
        from modules.vision.camera_manager import CameraManager

        config = _load_config()
        manager = CameraManager(config=config)
        await manager.load()

        rooms = manager.get_available_rooms()
        ok(f"Available rooms with cameras: {rooms}")

        frame = None
        if rooms:
            frame = await manager.capture_frame_async(rooms[0])
            if frame is not None:
                ok(f"Captured frame from '{rooms[0]}': shape={frame.shape}")
            else:
                info("Camera returned None — using synthetic frame")
        else:
            info("No cameras configured — using synthetic frame")

        await manager.close()

        if frame is None:
            frame = _make_synthetic_frame()
            ok(f"Synthetic frame created: shape={frame.shape}")

        return True, frame

    except Exception as e:
        fail(f"Camera manager error: {e}")
        return False, _make_synthetic_frame()


async def test_light_detector(frame: np.ndarray) -> bool:
    print(f"\n{BOLD}[2] Light Detector{RESET}")
    try:
        from modules.vision.light_detector import LightDetector

        config = _load_config()
        detector = LightDetector(config=config)

        # BUG FIX: analyze_async() returns Optional[bool] not a dict.
        # Cannot use "lights_on" in result — result IS the lights_on bool.
        lights_on = await detector.analyze_async(frame)
        ok(f"Light analysis: lights_on={lights_on}")
        assert lights_on is None or isinstance(lights_on, bool), \
            f"Expected Optional[bool], got {type(lights_on)}"
        return True

    except Exception as e:
        fail(f"Light detector error: {e}")
        return False


async def test_posture_analyzer(frame: np.ndarray) -> bool:
    print(f"\n{BOLD}[3] Posture Analyzer (MediaPipe){RESET}")
    try:
        from modules.vision.posture_analyzer import PostureAnalyzer

        config = _load_config()
        # BUG FIX: PostureAnalyzer takes config: dict
        analyzer = PostureAnalyzer(config)
        await analyzer.load_async()

        t0 = time.time()
        result = await analyzer.analyze_async(frame)
        elapsed = time.time() - t0

        ok(f"Posture result: '{result}' in {elapsed*1000:.0f}ms")
        assert result in ("standing", "sitting", "lying", "unknown"), \
            f"Unexpected posture: {result}"
        return True

    except Exception as e:
        fail(f"Posture analyzer error: {e}")
        return False


async def test_object_detector(frame: np.ndarray) -> bool:
    print(f"\n{BOLD}[4] Object Detector (YOLOv8){RESET}")
    try:
        from modules.vision.object_detector import ObjectDetector

        config = _load_config()
        # BUG FIX: ObjectDetector takes config: dict
        detector = ObjectDetector(config)
        await detector.load_async()

        t0 = time.time()
        detections = await detector.detect_async(frame)
        elapsed = time.time() - t0

        summary = detector.summarize(detections)
        ok(f"Detection complete in {elapsed*1000:.0f}ms: {len(detections)} objects")
        ok(f"Summary: {summary}")
        return True

    except Exception as e:
        fail(f"Object detector error: {e}")
        return False


async def test_scene_analyzer(frame: np.ndarray) -> bool:
    print(f"\n{BOLD}[5] Scene Analyzer (Configured Ollama Vision Model){RESET}")
    try:
        from modules.vision.scene_analyzer import SceneAnalyzer

        config = _load_config()
        # BUG FIX: SceneAnalyzer takes (config, llm) — not flat model/base_url kwargs.
        # We create a throwaway OllamaLLM instance for the test.
        from modules.brain.llm import OllamaLLM
        llm = OllamaLLM(config)
        analyzer = SceneAnalyzer(config=config, llm=llm)

        t0 = time.time()
        description = await analyzer.describe_async(
            frame,
            room="office",
            objects=[{"class": "test frame", "confidence": 1.0, "label": "test frame"}],
        )
        elapsed = time.time() - t0

        if description:
            ok(f"Scene description in {elapsed:.1f}s: {description!r}")
        else:
            vision_model = config.get("ollama", {}).get("vision_model", config.get("ollama", {}).get("model", ""))
            info(f"Scene analyzer returned empty — configured vision model may not be loaded: {vision_model}")
            if vision_model:
                info(f"Run: ollama pull {vision_model}")

        return True  # Not fatal — vision is optional during bring-up

    except Exception as e:
        fail(f"Scene analyzer error: {e}")
        return False


def print_summary(results: dict) -> None:
    print(f"\n{'=' * 50}")
    print(f"{BOLD}  VISION PIPELINE TEST SUMMARY{RESET}")
    print(f"{'=' * 50}")
    for check, passed in results.items():
        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"  {status}  {check}")
    print(f"{'=' * 50}")
    if all(results.values()):
        print(f"\n{GREEN}{BOLD}  [OK] All vision tests passed.{RESET}\n")
    else:
        failed = sum(1 for v in results.values() if not v)
        print(f"\n{RED}{BOLD}  [X] {failed} test(s) failed.{RESET}\n")


async def run_tests() -> int:
    cam_ok, frame = await test_camera_manager()

    results = {
        "Camera Manager":    cam_ok,
        "Light Detector":    await test_light_detector(frame),
        "Posture Analyzer":  await test_posture_analyzer(frame),
        "Object Detector":   await test_object_detector(frame),
        "Scene Analyzer":    await test_scene_analyzer(frame),
    }

    print_summary(results)
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(run_tests()))
