"""
JARVIS — recognition diagnostic (read-only)
===========================================
Answers the question the recognition audit could not answer from code
alone: WHY do Cole and Anna keep landing in "unknown faces"?

It reads data/jarvis.db read-only and reports:

  * per-person face/voice sample counts — TOTAL vs LIVE (the matcher only
    loads arcface_buffalo_l_v1 512-dim rows; legacy facenet rows are
    silently excluded, so "540 in the DB" can be far fewer live);
  * within-person cohesion — how tight each person's own bank is;
  * cross-person separation — the MAX nearest-neighbour cosine between
    two people's banks (the matcher is NN, so the max is what trips the
    margin gate — high Cole<->Anna here explains the symptom directly);
  * a contamination hunt — samples that score higher against a DIFFERENT
    person than their own (likely mislabeled, and poison for NN matching);
  * the pending queue, and a re-score of unresolved face rows against the
    live bank: would they MATCH / be AMBIGUOUS / stay UNKNOWN — i.e. is
    the failure a recall miss (Mode A) or a margin trip (Mode B)?

Run:  .venv\\Scripts\\python.exe scripts\\diagnose_recognition.py
Read-only — safe to run any time. ASCII-only output.
"""

from __future__ import annotations

import sqlite3
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "jarvis.db"

# Mirrors modules/identity/identity_manager.py.
ACTIVE_FACE_MODEL = "arcface_buffalo_l_v1"
FACE_DIM = 512
FACE_MATCH = 0.50
FACE_MARGIN = 0.10
FACE_STRANGER = 0.35
VOICE_DIM = 256
FACE_PENDING_KINDS = ("pending_cluster_face", "face_drift")


def _norm(vec: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    return vec / n if n > 0 else vec


def _load_face_bank(db: sqlite3.Connection):
    """Return (persons, total, mats, by_model, wrong_dim, src_mats).
    mats: pid -> Nx512 normalized matrix the matcher would load.
    src_mats: (pid, source) -> matrix, for the per-source cohesion view."""
    persons = {r["id"]: r["name"]
               for r in db.execute("SELECT id, name FROM persons")}
    total = {pid: 0 for pid in persons}
    live: dict[int, list[np.ndarray]] = {pid: [] for pid in persons}
    src_live: dict[tuple, list[np.ndarray]] = {}
    by_model: dict[str, int] = {}
    wrong_dim = 0
    for r in db.execute(
        "SELECT person_id, embedding, model_version, source FROM face_samples"
    ):
        pid = r["person_id"]
        mv = r["model_version"] or "<null>"
        by_model[mv] = by_model.get(mv, 0) + 1
        if pid in total:
            total[pid] += 1
        emb = np.frombuffer(r["embedding"], dtype=np.float32)
        if mv != ACTIVE_FACE_MODEL:
            continue
        if emb.size != FACE_DIM:
            wrong_dim += 1
            continue
        if pid in live:
            normed = _norm(emb.astype(np.float32))
            live[pid].append(normed)
            src_live.setdefault(
                (pid, r["source"] or "<null>"), []
            ).append(normed)
    mats = {pid: (np.stack(v) if v else np.zeros((0, FACE_DIM), np.float32))
            for pid, v in live.items()}
    src_mats = {k: np.stack(v) for k, v in src_live.items()}
    return persons, total, mats, by_model, wrong_dim, src_mats


def _within_cohesion(mat: np.ndarray) -> float:
    """Mean off-diagonal cosine within one person's bank."""
    if mat.shape[0] < 2:
        return float("nan")
    sims = mat @ mat.T
    n = mat.shape[0]
    off = (sims.sum() - np.trace(sims)) / (n * n - n)
    return float(off)


def _best_against(mat: np.ndarray, q: np.ndarray) -> float:
    """Max cosine of q against a bank (the NN score the matcher uses)."""
    if mat.shape[0] == 0:
        return -1.0
    return float(np.max(mat @ q))


def main() -> int:
    if not DB_PATH.exists():
        print(f"[ERR] {DB_PATH} not found")
        return 1
    db = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row

    print("JARVIS RECOGNITION DIAGNOSTIC")
    print("=" * 60)
    print(f"DB: {DB_PATH}\n")

    persons, total, mats, by_model, wrong_dim, src_mats = _load_face_bank(db)

    # ── persons + face bank ──────────────────────────────────────────────
    print(f"PERSONS ({len(persons)})")
    for pid, name in sorted(persons.items()):
        vc = db.execute(
            "SELECT COUNT(*) FROM voice_samples WHERE person_id=?", (pid,)
        ).fetchone()[0]
        print(f"  [{pid}] {name:<12} face: {total.get(pid,0):>4} total / "
              f"{mats[pid].shape[0]:>4} live   voice: {vc}")
    print()

    print("FACE BANK -- model-version breakdown (all face_samples rows)")
    for mv, n in sorted(by_model.items(), key=lambda x: -x[1]):
        tag = "  <- LIVE" if mv == ACTIVE_FACE_MODEL else "  <- EXCLUDED from matching"
        print(f"  {mv:<28} {n:>5}{tag}")
    if wrong_dim:
        print(f"  (+ {wrong_dim} rows tagged ArcFace but NOT {FACE_DIM}-dim "
              f"-- corrupt, excluded)")
    live_total = sum(m.shape[0] for m in mats.values())
    all_total = sum(total.values())
    print(f"  => {live_total} of {all_total} face samples actually reach "
          f"the matcher.")
    print()

    # ── per-person cohesion ──────────────────────────────────────────────
    print("FACE BANK -- per-person health")
    for pid, name in sorted(persons.items()):
        m = mats[pid]
        if m.shape[0] == 0:
            print(f"  {name:<12} live=0  -- CANNOT BE RECOGNIZED (empty bank)")
            continue
        coh = _within_cohesion(m)
        coh_s = f"{coh:.3f}" if coh == coh else "n/a (1 sample)"
        flag = "  <- LOW, bank is scattered" if (coh == coh and coh < 0.45) else ""
        print(f"  {name:<12} live={m.shape[0]:>4}  within-cohesion={coh_s}{flag}")
    print()

    # ── cohesion by capture source ───────────────────────────────────────
    print("FACE BANK -- cohesion by capture source (where the junk is)")
    print("  (good same-person ArcFace samples cohere ~0.5+; near 0 = junk)")
    for (pid, source), m in sorted(
        src_mats.items(),
        key=lambda kv: (persons.get(kv[0][0], ""), kv[0][1]),
    ):
        coh = _within_cohesion(m)
        coh_s = f"{coh:.3f}" if coh == coh else "n/a (1 sample)"
        flag = ""
        if coh == coh:
            flag = "  <- JUNK" if coh < 0.30 else (
                "  <- ok" if coh >= 0.45 else "")
        print(f"  {persons.get(pid,'?'):<10} / {source:<14} "
              f"n={m.shape[0]:>4}  cohesion={coh_s}{flag}")
    print()

    # ── cross-person separation ──────────────────────────────────────────
    print("CROSS-PERSON SEPARATION -- max NN cosine between banks")
    print("  (the matcher is nearest-neighbour; a high max means a live")
    print("   face can score close to BOTH people -> margin gate trips)")
    pairs = []
    for a, b in combinations(sorted(persons), 2):
        ma, mb = mats[a], mats[b]
        if ma.shape[0] == 0 or mb.shape[0] == 0:
            continue
        mx = float(np.max(ma @ mb.T))
        pairs.append((mx, persons[a], persons[b]))
    for mx, na, nb in sorted(pairs, reverse=True):
        flag = "  <- ABOVE match threshold; margin-trip risk" if mx >= FACE_MATCH else ""
        print(f"  {na} <-> {nb}: max={mx:.3f}{flag}")
    if not pairs:
        print("  (need 2+ people with live banks)")
    print()

    # ── contamination hunt ───────────────────────────────────────────────
    print("CONTAMINATION HUNT -- samples matching another person better")
    print("  than their own bank (likely mislabeled; poison for NN match)")
    suspects = 0
    for pid, name in sorted(persons.items()):
        m = mats[pid]
        for i in range(m.shape[0]):
            q = m[i]
            # best against own bank, excluding self
            own = m @ q
            own[i] = -1.0
            own_best = float(np.max(own)) if m.shape[0] > 1 else -1.0
            other_best, other_name = -1.0, None
            for opid, oname in persons.items():
                if opid == pid:
                    continue
                ob = _best_against(mats[opid], q)
                if ob > other_best:
                    other_best, other_name = ob, oname
            if other_best > own_best and other_best >= FACE_MATCH:
                suspects += 1
                if suspects <= 12:
                    print(f"  {name} sample #{i}: own_best={own_best:.3f}  "
                          f"{other_name}_best={other_best:.3f}  <- SUSPECT")
    if suspects == 0:
        print("  none found -- no obvious cross-person mislabeling.")
    elif suspects > 12:
        print(f"  ... and {suspects - 12} more ({suspects} total).")
    else:
        print(f"  {suspects} suspect sample(s).")
    print()

    # ── pending queue ────────────────────────────────────────────────────
    print("PENDING QUEUE")
    rows = db.execute(
        "SELECT kind, resolved, COUNT(*) n FROM identity_pending "
        "GROUP BY kind, resolved"
    ).fetchall()
    unresolved_total = 0
    for r in rows:
        state = {0: "unresolved", 1: "applied", 2: "rejected"}.get(
            r["resolved"], str(r["resolved"]))
        if r["resolved"] == 0:
            unresolved_total += r["n"]
        print(f"  {r['kind']:<24} {state:<11} {r['n']:>5}")
    print(f"  => {unresolved_total} unresolved rows total")
    print()

    # ── pending face re-score: Mode A vs Mode B ──────────────────────────
    print("PENDING FACE RE-SCORE -- unresolved face rows vs the live bank")
    qmarks = ",".join("?" * len(FACE_PENDING_KINDS))
    pend = db.execute(
        f"SELECT embedding FROM identity_pending "
        f"WHERE resolved=0 AND kind IN ({qmarks})",
        FACE_PENDING_KINDS,
    ).fetchall()
    would_match = would_ambig = still_unknown = stale_dim = 0
    for r in pend:
        emb = np.frombuffer(r["embedding"], dtype=np.float32)
        if emb.size != FACE_DIM:
            stale_dim += 1
            continue
        q = _norm(emb.astype(np.float32))
        scored = sorted(
            (_best_against(mats[pid], q) for pid in persons if mats[pid].shape[0]),
            reverse=True,
        )
        if not scored:
            still_unknown += 1
            continue
        best = scored[0]
        second = scored[1] if len(scored) > 1 else -1.0
        if best < FACE_MATCH:
            still_unknown += 1
        elif (best - second) < FACE_MARGIN:
            would_ambig += 1
        else:
            would_match += 1
    print(f"  scored {len(pend) - stale_dim} unresolved face rows "
          f"({stale_dim} skipped: non-{FACE_DIM}-dim/stale)")
    print(f"  would MATCH cleanly  (>={FACE_MATCH}, margin ok):   {would_match:>5}"
          f"   <- recoverable: bank just needs these")
    print(f"  would be AMBIGUOUS   (>={FACE_MATCH}, margin <{FACE_MARGIN}): {would_ambig:>5}"
          f"   <- Mode B: margin gate")
    print(f"  still UNKNOWN        (<{FACE_MATCH}):               {still_unknown:>5}"
          f"   <- Mode A: recall miss")
    dom = max(
        ("Mode A (recall miss -- bank too sparse/poor)", still_unknown),
        ("Mode B (margin trip -- people not separable)", would_ambig),
        ("recoverable (would match now)", would_match),
        key=lambda x: x[1],
    )
    print(f"  => dominant: {dom[0]}")
    print()
    print("=" * 60)
    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
