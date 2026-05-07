"""
JARVIS — Ambient Home AI
========================
Mission: Verification script for the persona system. Doesn't touch the live
         orchestrator — instantiates PersonaManager directly with the
         project's config, then exercises every safety floor and state
         transition described in PERSONA_BOOTSTRAP.md §7.

         Each test prints PASS/FAIL inline. Exit code 0 = all pass; 1 = any
         fail. Run after touching anything in modules/brain/persona_manager.py
         or core/config.py's persona schema.

Usage:
    python scripts/test_personas.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

# Make the project importable regardless of cwd
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from core.config import expand_and_validate  # noqa: E402
from core.exceptions import PersonaError  # noqa: E402
from modules.brain.persona_manager import PersonaManager, RoomOccupancy  # noqa: E402


# Make the env vars happy for env-expansion of the wyze URLs in config.yaml
for k in ("WYZE_RTSP_USER", "WYZE_RTSP_PASSWORD", "WYZE_SSH_USER",
          "WYZE_SSH_PASSWORD", "WYZE_SSH_KEY_PATH"):
    os.environ.setdefault(k, "test")


def _load_persona_config():
    import yaml
    with open(_REPO_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg, _ = expand_and_validate(raw)
    if "_typed_personas" not in cfg:
        raise SystemExit("FAIL: config.yaml has no personas section")
    return cfg


_results: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _results.append((name, cond, detail))
    marker = "PASS" if cond else "FAIL"
    print(f"  [{marker}] {name}{(' — ' + detail) if detail else ''}")


def _make_pm(cfg) -> PersonaManager:
    """Construct a fresh PersonaManager. No broadcast hook (dashboard
    isn't running in this test); transitions log to loguru only."""
    return PersonaManager(
        personas=cfg["_typed_personas"],
        overlay=cfg["_persona_overlay"],
        revert_cfg=cfg["_persona_revert_cfg"],
        broadcast=None,
    )


def _set_alone(pm: PersonaManager, room: str = "office") -> None:
    """Mark Cole alone in `room` so the privacy gate lets uwu activate."""
    pm.state().room_occupancy[room] = RoomOccupancy(
        person_count=1, cole_present=True, unknown=False, updated_at=time.time(),
    )


def _set_not_alone(pm: PersonaManager, room: str = "office") -> None:
    pm.state().room_occupancy[room] = RoomOccupancy(
        person_count=2, cole_present=True, unknown=False, updated_at=time.time(),
    )


async def t_config_loads():
    print("\n=== Config + visibility ===")
    cfg = _load_persona_config()
    personas = cfg["_typed_personas"]
    check("personas dict has 'default'", "default" in personas)
    check("personas dict has 'uwu'", "uwu" in personas)
    check("uwu is hidden", not personas["uwu"].visible_in_ui)
    check("uwu requires_privacy", personas["uwu"].requires_privacy)
    check("default visible", personas["default"].visible_in_ui)
    check("overlay non-empty", len(cfg["_persona_overlay"]) > 0)
    return cfg


async def t_visible_listing(cfg):
    print("\n=== Visible listing ===")
    pm = _make_pm(cfg)
    visible_names = {p["name"] for p in pm.list_visible()}
    check("uwu absent from visible list", "uwu" not in visible_names)
    check("default present in visible list", "default" in visible_names)
    return pm


async def t_privacy_gate(cfg):
    print("\n=== Privacy gate ===")
    pm = _make_pm(cfg)
    # Empty occupancy = unknown = fail closed
    raised = False
    try:
        await pm.set("uwu")
    except PersonaError:
        raised = True
    check("uwu blocked when occupancy unknown", raised)
    check("active still default after blocked set", pm.current_name() == "default")

    _set_alone(pm)
    await pm.set("uwu")
    check("uwu activates when alone", pm.current_name() == "uwu")
    await pm.revert(reason="test_cleanup")


async def t_force_bypass(cfg):
    print("\n=== Force bypass ===")
    pm = _make_pm(cfg)
    _set_not_alone(pm)
    raised = False
    try:
        await pm.set("uwu")
    except PersonaError:
        raised = True
    check("uwu blocked without force when not alone", raised)
    await pm.set("uwu", force=True)
    check("uwu activates with force=True", pm.current_name() == "uwu")


async def t_person_entry_revert(cfg):
    print("\n=== Person-entry revert (UNCONDITIONAL) ===")
    pm = _make_pm(cfg)
    _set_alone(pm)
    await pm.set("uwu", lock=True)
    check("uwu active + locked", pm.current_name() == "uwu" and pm.is_locked())
    # Foreign person enters
    await pm.notify_face_identified(room="office", identity="someone_else")
    check("revert fires even when locked", pm.current_name() == "default")

    # Cole's face should NOT trigger revert
    pm = _make_pm(cfg)
    _set_alone(pm)
    await pm.set("uwu")
    await pm.notify_face_identified(room="office", identity="cole")
    check("Cole's face does not revert", pm.current_name() == "uwu")
    await pm.revert(reason="test_cleanup")


async def t_overlay_in_prompt(cfg):
    print("\n=== Overlay always present ===")
    pm = _make_pm(cfg)
    # Default persona — overlay still applies
    p = pm.composed_system_prompt()
    check("default prompt contains PRIVACY DIRECTIVE", "PRIVACY DIRECTIVE" in p)
    check("default prompt contains 'Jarvis'", "Jarvis" in p)
    # Switch to uwu — overlay still present
    _set_alone(pm)
    await pm.set("uwu")
    p2 = pm.composed_system_prompt()
    check("uwu prompt contains overlay", "PRIVACY DIRECTIVE" in p2)
    check("uwu prompt contains uwu IDENTITY block", "IDENTITY" in p2)
    await pm.revert(reason="test_cleanup")


async def t_output_filter(cfg):
    print("\n=== Output filter (defense in depth) ===")
    pm = _make_pm(cfg)
    # Default + not alone + leak in text → scrub
    _set_not_alone(pm)
    leak = "Switching back to uwu mode soon."
    scrubbed = pm.filter_output(leak)
    check("uwu mention scrubbed when not alone", "uwu" not in scrubbed.lower())

    # Default + alone → no scrub (Cole can see his own modes)
    _set_alone(pm)
    same = pm.filter_output(leak)
    check("uwu mention preserved when alone (Cole only)", "uwu" in same.lower())

    # uwu active → no scrub even if not alone (already in private mode;
    # filter is for default-mode leaks specifically)
    await pm.set("uwu")
    _set_not_alone(pm)
    same2 = pm.filter_output(leak)
    check("uwu mention preserved when uwu is active", "uwu" in same2.lower())
    await pm.revert(reason="test_cleanup")


async def t_phone_call(cfg):
    print("\n=== Phone-call revert + resume ===")
    pm = _make_pm(cfg)
    _set_alone(pm)
    await pm.set("uwu")
    await pm.notify_phone_call_started()
    check("phone-start reverts uwu", pm.current_name() == "default")
    check("pending_resume saved as uwu", pm.state().pending_resume == "uwu")
    # Call ends, still alone — manager broadcasts (no-op without hook),
    # accept_pending_resume reactivates.
    _set_alone(pm)
    await pm.notify_phone_call_ended()
    ok = await pm.accept_pending_resume()
    check("resume after call succeeds", ok and pm.current_name() == "uwu")
    await pm.revert(reason="test_cleanup")


async def t_lock_blocks_phone_revert(cfg):
    print("\n=== Lock blocks phone but NOT person-entry ===")
    pm = _make_pm(cfg)
    _set_alone(pm)
    await pm.set("uwu", lock=True)
    await pm.notify_phone_call_started()
    check("phone revert blocked by lock", pm.current_name() == "uwu")
    # Person entry still wins
    await pm.notify_face_identified(room="office", identity="stranger")
    check("person-entry still reverts despite lock", pm.current_name() == "default")


async def main() -> int:
    cfg = await t_config_loads()
    await t_visible_listing(cfg)
    await t_privacy_gate(cfg)
    await t_force_bypass(cfg)
    await t_person_entry_revert(cfg)
    await t_overlay_in_prompt(cfg)
    await t_output_filter(cfg)
    await t_phone_call(cfg)
    await t_lock_blocks_phone_revert(cfg)

    fails = [(n, d) for n, ok, d in _results if not ok]
    print(f"\n{'=' * 60}")
    print(f"  {len(_results) - len(fails)}/{len(_results)} checks passed")
    if fails:
        print("\nFailed checks:")
        for n, d in fails:
            print(f"  - {n}{(' — ' + d) if d else ''}")
    print(f"{'=' * 60}\n")
    return 0 if not fails else 1


if __name__ == "__main__":
    if sys.platform == "win32" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    sys.exit(asyncio.run(main()))
