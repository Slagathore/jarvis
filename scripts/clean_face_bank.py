"""
JARVIS — face-bank cleaner (the bank rebuild)
=============================================
The recognition diagnostic showed the face bank is incoherent — within-
person cohesion ~0.10 where healthy ArcFace is ~0.5+ — because hundreds
of low-quality `live_question` samples (pending clusters bulk-confirmed
in the dashboard) drowned the real faces.

This tool rebuilds each person's bank around its COHERENT CORE:

  * for every person, rank their live ArcFace samples by centrality —
    mean cosine to that person's other samples;
  * the high-centrality samples ARE the person's real face; the low ones
    are junk / misattributions dragging cohesion (and recognition) down;
  * keep the core (down to a MIN_KEEP floor, up to a MAX_KEEP cap),
    quarantine the rest.

Quarantine is REVERSIBLE and non-destructive: the sample's model_version
is re-tagged to '<active>_quarantined', so IdentityManager's loader
(which filters to the exact active tag) simply stops loading it. The row
and its crop stay in the DB. Undo: re-tag back.

Usage:
  .venv\\Scripts\\python.exe scripts\\clean_face_bank.py            # dry run — report only
  .venv\\Scripts\\python.exe scripts\\clean_face_bank.py --apply    # quarantine the junk
  .venv\\Scripts\\python.exe scripts\\clean_face_bank.py --undo     # un-quarantine everything

ASCII-only output.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "jarvis.db"
ACTIVE = "arcface_buffalo_l_v1"
QUARANTINE_TAG = "arcface_buffalo_l_v1_quarantined"
DIM = 512

CORE_FLOOR = 0.32   # centrality to count as "core" — a sample this close
                    # to the person's others is plausibly really them
MIN_KEEP = 15       # never quarantine a person below this many samples
MAX_KEEP = 60       # cap (matches IdentityManager SAMPLES_PER_PERSON_MAX)


def _norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _cohesion(mat: np.ndarray) -> float:
    if mat.shape[0] < 2:
        return float("nan")
    sims = mat @ mat.T
    n = mat.shape[0]
    return float((sims.sum() - np.trace(sims)) / (n * n - n))


def _plan(db: sqlite3.Connection) -> list[dict]:
    """Per person, decide which sample ids to keep vs quarantine."""
    persons = {r["id"]: r["name"]
               for r in db.execute("SELECT id, name FROM persons")}
    plans: list[dict] = []
    for pid, name in sorted(persons.items()):
        rows = db.execute(
            "SELECT id, embedding FROM face_samples "
            "WHERE person_id=? AND model_version=?",
            (pid, ACTIVE),
        ).fetchall()
        ids, vecs = [], []
        for r in rows:
            emb = np.frombuffer(r["embedding"], dtype=np.float32)
            if emb.size == DIM:
                ids.append(int(r["id"]))
                vecs.append(_norm(emb.astype(np.float32)))
        n = len(ids)
        if n == 0:
            continue
        if n <= MIN_KEEP:
            plans.append({"pid": pid, "name": name, "total": n,
                          "keep": ids, "quarantine": [],
                          "before": _cohesion(np.stack(vecs)) if n > 1 else float("nan"),
                          "after": _cohesion(np.stack(vecs)) if n > 1 else float("nan")})
            continue
        mat = np.stack(vecs)
        sims = mat @ mat.T
        # centrality = mean cosine to this person's OTHER samples.
        centrality = (sims.sum(axis=1) - 1.0) / (n - 1)
        order = np.argsort(-centrality)          # most-central first
        core = [i for i in order if centrality[i] >= CORE_FLOOR]
        if len(core) < MIN_KEEP:
            core = list(order[:MIN_KEEP])        # best available
        if len(core) > MAX_KEEP:
            core = list(order[:MAX_KEEP])
        core_set = set(int(i) for i in core)
        keep = [ids[i] for i in range(n) if i in core_set]
        quar = [ids[i] for i in range(n) if i not in core_set]
        kept_mat = np.stack([vecs[i] for i in range(n) if i in core_set])
        plans.append({
            "pid": pid, "name": name, "total": n,
            "keep": keep, "quarantine": quar,
            "before": _cohesion(mat),
            "after": _cohesion(kept_mat),
        })
    return plans


def main() -> int:
    ap = argparse.ArgumentParser(description="Rebuild the face bank around its coherent core.")
    ap.add_argument("--apply", action="store_true",
                    help="quarantine the junk (reversible re-tag)")
    ap.add_argument("--undo", action="store_true",
                    help="un-quarantine every sample")
    args = ap.parse_args()

    if not DB_PATH.exists():
        print(f"[ERR] {DB_PATH} not found")
        return 1

    if args.undo:
        db = sqlite3.connect(DB_PATH)
        n = db.execute(
            "UPDATE face_samples SET model_version=? WHERE model_version=?",
            (ACTIVE, QUARANTINE_TAG),
        ).rowcount
        db.commit()
        db.close()
        print(f"[clean_face_bank] un-quarantined {n} sample(s).")
        return 0

    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    plans = _plan(db)

    print("JARVIS FACE-BANK CLEANER" + ("  [DRY RUN]" if not args.apply else "  [APPLY]"))
    print("=" * 64)
    total_q = 0
    for p in plans:
        before = f"{p['before']:.3f}" if p["before"] == p["before"] else "n/a"
        after = f"{p['after']:.3f}" if p["after"] == p["after"] else "n/a"
        total_q += len(p["quarantine"])
        print(f"  {p['name']:<12} total={p['total']:>4}  "
              f"keep={len(p['keep']):>3}  quarantine={len(p['quarantine']):>4}  "
              f"cohesion {before} -> {after}")
        if p["after"] == p["after"] and p["after"] < 0.30 and p["keep"]:
            print(f"               ^ still low after cleaning — "
                  f"{p['name']} should re-enroll from fresh frontal captures")
    print("-" * 64)
    print(f"  would quarantine {total_q} sample(s) across "
          f"{len(plans)} person(s)")

    if not args.apply:
        print("\nDry run — nothing changed. Re-run with --apply to quarantine,")
        print("or --undo to reverse a previous --apply.")
        db.close()
        return 0

    quar_ids = [sid for p in plans for sid in p["quarantine"]]
    for sid in quar_ids:
        db.execute(
            "UPDATE face_samples SET model_version=? WHERE id=?",
            (QUARANTINE_TAG, sid),
        )
    db.commit()
    db.close()
    print(f"\n[clean_face_bank] quarantined {len(quar_ids)} sample(s). "
          f"Reversible: scripts/clean_face_bank.py --undo")
    return 0


if __name__ == "__main__":
    sys.exit(main())
