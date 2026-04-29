"""
JARVIS — Ambient Home AI
========================
Mission: Phase 4 test script. Validates MQTT connectivity, topic routing, and
         ESP32 node manager. Tests publish/subscribe round-trips without
         requiring actual hardware nodes — uses loopback messaging.

         Run: python scripts/test_mqtt.py
         Requires: Mosquitto running on localhost:1883

Modules: scripts/test_mqtt.py
Classes: (none)
Functions:
    test_mqtt_connect()      — Connect to MQTT broker
    test_mqtt_pubsub()       — Publish a message and receive it back on same topic
    test_node_manager()      — Initialize NodeManager, check get_status_summary()
    test_node_heartbeat()    — Publish a fake node status, verify manager picks it up
    run_tests()              — Run all tests, print summary

#todo: Add test for ESP32-CAM MJPEG stream connection (if node is online)
#todo: Add test for audio routing (publish binary audio, verify receipt)
#todo: Add latency measurement for MQTT round-trip
"""

import asyncio
import os
import sys
import time
from typing import Any

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


def _configure_event_loop_policy() -> None:
    if sys.platform == "win32" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


async def _start_listener(client: Any) -> asyncio.Task:
    task = asyncio.create_task(client.listen_forever())
    await asyncio.sleep(0.25)
    return task


async def _stop_listener(task: asyncio.Task | None) -> None:
    if task is None:
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


async def test_mqtt_connect() -> tuple[bool, Any | None]:
    print(f"\n{BOLD}[1] MQTT Connect{RESET}")
    try:
        from modules.network.mqtt_client import MQTTClient

        config = _load_config()

        from core.event_bus import EventBus
        bus = EventBus()
        # BUG FIX: MQTTClient takes (config, event_bus) — not flat broker/port/etc kwargs
        client = MQTTClient(config=config, event_bus=bus)
        mcfg = config.get("mqtt", {})

        connected = await client.connect()
        if not connected:
            fail(f"Failed to connect to {mcfg.get('broker', 'localhost')}:{mcfg.get('port', 1883)}")
            return False, None
        ok(f"Connected to {mcfg.get('broker', 'localhost')}:{mcfg.get('port', 1883)}")
        return True, client

    except Exception as e:
        fail(f"MQTT connect error: {e}")
        return False, None


async def test_mqtt_pubsub(client: Any | None) -> bool:
    print(f"\n{BOLD}[2] MQTT Publish / Subscribe Round-Trip{RESET}")
    if client is None:
        fail("Skipped — no MQTT connection")
        return False

    try:
        received: list[dict] = []
        event = asyncio.Event()

        test_topic = "jarvis/test/ping"
        test_payload = {"msg": "hello", "ts": time.time()}

        async def handler(topic: str, data: dict):
            received.append(data)
            event.set()

        client.subscribe(test_topic, handler)
        listener_task = await _start_listener(client)
        try:
            await client.publish(test_topic, test_payload)

            try:
                await asyncio.wait_for(event.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                fail("No message received within 5 seconds (check Mosquitto logs)")
                return False

            ok(f"Round-trip successful: {received[0]}")
            return True
        finally:
            await _stop_listener(listener_task)

    except Exception as e:
        fail(f"Pub/sub error: {e}")
        return False


async def test_node_manager(client: Any | None) -> bool:
    print(f"\n{BOLD}[3] Node Manager{RESET}")
    if client is None:
        fail("Skipped — no MQTT connection")
        return False
    try:
        from modules.network.node_manager import NodeManager

        config = _load_config()
        # BUG FIX: NodeManager param is mqtt_client not mqtt
        manager = NodeManager(config=config, mqtt_client=client)
        await manager.load()

        summary = manager.get_status_summary()
        ok(f"Node manager loaded. Status summary: {summary}")

        online_rooms = manager.get_online_rooms()
        info(f"Online rooms: {online_rooms or 'none (hardware not deployed)'}")
        return True

    except Exception as e:
        fail(f"Node manager error: {e}")
        return False


async def test_node_heartbeat(client: Any | None) -> bool:
    print(f"\n{BOLD}[4] Node Heartbeat Simulation{RESET}")
    if client is None:
        fail("Skipped — no MQTT connection")
        return False

    try:
        from modules.network.node_manager import NodeManager

        config = _load_config()
        # BUG FIX: NodeManager param is mqtt_client not mqtt
        manager = NodeManager(config=config, mqtt_client=client)
        await manager.load()
        listener_task = await _start_listener(client)

        try:
            # Simulate an office node coming online
            await client.publish(
                "jarvis/nodes/office/status",
                {"status": "online", "ip": "192.168.1.101", "rssi": -55},
            )

            for _ in range(20):
                if manager.is_online("office"):
                    break
                await asyncio.sleep(0.1)

            is_online = manager.is_online("office")
            ok(f"Office node online after simulated heartbeat: {is_online}")
            assert is_online, "Expected simulated heartbeat to mark office online"
            return True
        finally:
            await _stop_listener(listener_task)

    except Exception as e:
        fail(f"Node heartbeat test error: {e}")
        return False


def print_summary(results: dict) -> None:
    print(f"\n{'=' * 50}")
    print(f"{BOLD}  MQTT / NETWORK TEST SUMMARY{RESET}")
    print(f"{'=' * 50}")
    for check, passed in results.items():
        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"  {status}  {check}")
    print(f"{'=' * 50}")
    if all(results.values()):
        print(f"\n{GREEN}{BOLD}  [OK] All MQTT tests passed.{RESET}\n")
    else:
        failed = sum(1 for v in results.values() if not v)
        print(f"\n{RED}{BOLD}  [X] {failed} test(s) failed.{RESET}\n")


async def run_tests() -> int:
    conn_ok, client = await test_mqtt_connect()

    results = {
        "MQTT Connect":        conn_ok,
        "Pub/Sub Round-Trip":  await test_mqtt_pubsub(client),
        "Node Manager":        await test_node_manager(client),
        "Node Heartbeat":      await test_node_heartbeat(client),
    }

    if client:
        try:
            await client.disconnect()
        except Exception:
            pass

    print_summary(results)
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    _configure_event_loop_policy()
    sys.exit(asyncio.run(run_tests()))
