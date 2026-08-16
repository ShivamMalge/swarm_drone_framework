"""
Copy generated plots into the manuscript figure slots and print md5 pairs.

The manuscript embeds `manuscript/figN.png`, but the plotting pipeline writes
descriptive names into `plots/`. That mapping used to be a manual copy, which
is how a fabricated figure survived the fix to the code that produced it: the
plotting code was repaired, `plots/` was regenerated, and `manuscript/` was
never updated. Run this after `plot_results.py` and check the printed hashes.

Usage:
    python experiments/plot_results.py
    python experiments/publish_figures.py            # copy + verify
    python experiments/publish_figures.py --check    # verify only, non-zero exit on drift
"""

from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLOTS = ROOT / "plots"
MANUSCRIPT = ROOT / "manuscript"

# plots/<source>.png  ->  manuscript/<dest>.png
FIGURE_MAP = {
    "kinematic_stability": "fig1",
    "spectral_stability": "fig2",
    "thermodynamic_decay": "fig3",
}


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main(check_only: bool = False) -> int:
    drift = 0
    print(f"{'figure':<26} {'source md5':<34} {'manuscript md5':<34} status")
    print("-" * 108)

    for source, dest in FIGURE_MAP.items():
        src_path = PLOTS / f"{source}.png"
        dst_path = MANUSCRIPT / f"{dest}.png"

        if not src_path.exists():
            print(f"{dest + '.png':<26} MISSING SOURCE: {src_path}")
            drift += 1
            continue

        if not check_only:
            shutil.copyfile(src_path, dst_path)

        src_hash = md5(src_path)
        dst_hash = md5(dst_path) if dst_path.exists() else "<absent>"
        ok = src_hash == dst_hash
        drift += 0 if ok else 1
        print(f"{dest + '.png':<26} {src_hash:<34} {dst_hash:<34} {'MATCH' if ok else 'DRIFT'}")

    print("-" * 108)
    if drift:
        print(f"FAIL: {drift} figure(s) out of sync. The manuscript is not showing current data.")
    else:
        print("OK: every manuscript figure is byte-identical to its regenerated source.")
    return 1 if drift else 0


if __name__ == "__main__":
    sys.exit(main(check_only="--check" in sys.argv))
