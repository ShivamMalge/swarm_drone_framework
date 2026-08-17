"""
Encoding-agnostic sweep for retired claims and numbers.

Why this exists: three fabrication-era result files sat at REPO ROOT carrying
1999/527/0.29 in UTF-16 -- every character NUL-separated -- and were invisible
to every plain-text grep this project ran, including sweeps accepted as
verification evidence. The blind spot was encoding, not location. This tool
searches raw BYTES for each pattern in multiple encodings, so a hit cannot
hide behind a codec, and it opens zip containers (docx/xlsx) to scan their
members.

WHAT THE SWEEP COVERS
  - every file under the repo except .git/ and __pycache__/
  - patterns matched as bytes in: UTF-8/ASCII, UTF-16LE, UTF-16BE, UTF-32LE
  - zip containers (.docx, .xlsx, .zip): each member scanned the same way
  - .pyc, images, PDFs: raw byte scan (catches embedded literal strings)

WHAT THE SWEEP CANNOT REACH (stated, so the guarantee has known limits)
  - compressed payloads: PDF FlateDecode streams, PNG IDAT -- text that exists
    only inside a compressed stream will not match. The repo's PDFs are
    third-party papers (refrences/) plus draft PDFs slated for recompile; the
    PNGs are generated figures whose sources are the swept CSVs.
  - text rendered INTO images (figure pixels, architecture JPGs)
  - git history (intentionally: history is the audit trail, not the product)

Exit code: 0 if every hit is on the allowlist, 1 otherwise.
"""

import sys
import zipfile
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Retired claims. (pattern, why it is retired)
PATTERNS = [
    ("1999", "fabrication-era survival figure"),
    ("527", "fabrication-era baseline survival"),
    ("0.29", "corpse-inflated lambda_2"),
    ("285000", "never-existed event count"),
    ("15 million", "overstated event count"),
    ("+0.04", "fabricated proxy bias"),
    ("perfectly matches", "oracle-matching claim"),
    ("thermodynamic braking", "retired mechanism name"),
    ("4.5x", "superseded energy ratio"),
    ("4.5\\times", "superseded energy ratio (LaTeX)"),
]
# "Centralized Oracle" is swept but the CURRENT manuscript row label and its
# supporting prose legitimately use the term; those hits are allowlisted below.
PATTERNS.append(("Centralized Oracle", "term survives only as the relabelled arm"))

ENCODINGS = ["utf-8", "utf-16-le", "utf-16-be", "utf-32-le"]

# (path-suffix, pattern) pairs that are intentional. Each entry must say why.
ALLOWLIST = {
    # The audit trail quotes history on purpose; that is its job.
    "audit_findings.md": "*",
    "fixes_phases.md": "*",
    "rust_conversion_plan.md": "*",
    "STATUS.md": "*",
    # Third-party literature (years like 1999, page numbers like 527).
    "refrences/": "*",
    "Review/": "*",
    "Documents/": "*",
    # Current manuscript: "Centralized Oracle" is the live (relabelled) arm
    # name; "0.29" appears only inside explicitly-marked correction narratives
    # -- verified per-hit below rather than blanket-allowed.
    # "+0.04" and the LaTeX "4.5\times" appear ONLY inside the paper's
    # explicit withdrawal/correction narratives ("We therefore withdraw the
    # +0.04 figure..."; "an earlier revision reported ... The change is a
    # correction of the baseline"). Quoting a retired figure in the sentence
    # that retires it is the one place it belongs.
    "manuscript/final_manuscript.tex": {"Centralized Oracle", "+0.04", "4.5\\times"},
    # This tool and its tests name the patterns it hunts.
    "experiments/sweep_retired_claims.py": "*",
    # The F-09 explanatory comment quotes the retired figure on purpose
    # ("1999 +/- 2 read as a measurement when it is 3 idle survivors").
    "experiments/run_monte_carlo_table.py": {"1999"},
    # Live usage of the current (relabelled) arm name in run instructions.
    "README.md": {"Centralized Oracle"},
    # Regression-pin docstrings quote the incidents they guard against.
    "tests/test_regression_pins.py": "*",
    # Generated telemetry exports: numeric fields (event counters, etc.) can
    # coincide with retired figures. The pre-audit April export was deleted
    # outright; future exports land here and are data, not claims.
    "outputs/": "*",
    # Stale compiled draft, slated for recompile at the Phase 5 gate; its
    # numbers are the reason the gate requires recompilation.
    "draft 11.pdf": "*",
}

SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", ".pytest_cache"}


def allowed(rel: str, pattern: str) -> bool:
    for key, pats in ALLOWLIST.items():
        if rel.replace("\\", "/").startswith(key) or rel.replace("\\", "/").endswith(key):
            return pats == "*" or pattern in pats
    return False


def _digit_bounded(data: bytes, idx: int, needle_len: int, enc: str,
                   pattern: str) -> bool:
    """
    For patterns that BEGIN or END with a digit, reject matches embedded in a
    longer number: '527' must not match inside '736527047', and '0.29' must
    not match the prefix of '0.29598...'. Boundary checks are done at the
    encoded-character width (a UTF-16 digit is two bytes), which the first
    version of this tool got wrong by checking single bytes.
    """
    width = len("0".encode(enc))
    digits = {str(d).encode(enc) for d in range(10)}
    if pattern[0].isdigit():
        before = data[max(0, idx - width):idx]
        if before in digits:
            return False
    if pattern[-1].isdigit():
        after = data[idx + needle_len: idx + needle_len + width]
        if after in digits:
            return False
    return True


def scan_bytes(data: bytes, origin: str, hits: list) -> None:
    for pattern, why in PATTERNS:
        for enc in ENCODINGS:
            needle = pattern.encode(enc)
            found = False
            start = 0
            while True:
                idx = data.find(needle, start)
                if idx == -1:
                    break
                if _digit_bounded(data, idx, len(needle), enc, pattern):
                    found = True
                    break
                start = idx + 1
            if found:
                lo = max(0, idx - 40)
                hi = min(len(data), idx + len(needle) + 40)
                try:
                    ctx = data[lo:hi].decode(enc, errors="replace")
                except Exception:
                    ctx = repr(data[lo:hi])
                ctx = " ".join(ctx.split())
                hits.append((origin, pattern, enc, ctx))
                break  # one encoding hit per pattern per file is enough


def main() -> int:
    hits: list = []
    files = bytes_total = members = 0

    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        rel = str(path.relative_to(ROOT))
        data = path.read_bytes()
        files += 1
        bytes_total += len(data)
        scan_bytes(data, rel, hits)

        if zipfile.is_zipfile(BytesIO(data)):
            try:
                with zipfile.ZipFile(BytesIO(data)) as z:
                    for name in z.namelist():
                        members += 1
                        scan_bytes(z.read(name), f"{rel}::{name}", hits)
            except Exception as e:  # corrupt member: report, do not hide
                hits.append((rel, "<unreadable zip member>", "-", str(e)))

    flagged = [(o, p, e, c) for (o, p, e, c) in hits if not allowed(o, p)]
    allowed_hits = [(o, p, e, c) for (o, p, e, c) in hits if allowed(o, p)]

    print(f"Encoding-agnostic retired-claim sweep")
    print(f"  scanned : {files} files, {bytes_total/1e6:.1f} MB, "
          f"{members} zip members")
    print(f"  encodings per pattern: {', '.join(ENCODINGS)}")
    print(f"  hits: {len(allowed_hits)} allowlisted, {len(flagged)} FLAGGED\n")

    if allowed_hits:
        print("Allowlisted (intentional history / third-party / live label):")
        seen = set()
        for o, p, e, _ in allowed_hits:
            key = (o.split("::")[0].split("/")[0].split("\\")[0], p)
            if key not in seen:
                seen.add(key)
                print(f"  [{e:>9}] {p!r:26} in {o}")
        print()

    if flagged:
        print("FLAGGED -- retired material outside the allowlist:")
        for o, p, e, c in flagged:
            print(f"  [{e:>9}] {p!r} in {o}\n      ...{c}...")
        return 1

    print("CLEAN: no retired material outside the allowlist.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
