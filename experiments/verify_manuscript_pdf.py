"""
Post-recompile verification of manuscript/final_manuscript.pdf.

The F-07 drift (a false claim living only in the .tex, invisible in the stale
PDF) came from never checking compiled output against source expectations.
This does for the PDF what publish_figures.py does for figures: the compiled
artifact is verified, not assumed. The text layer is extracted with pypdf,
which reaches INSIDE the FlateDecode streams the byte-level sweep cannot.
"""
import re
import sys

from pypdf import PdfReader

PDF = "manuscript/final_manuscript.pdf"

reader = PdfReader(PDF)
text = "\n".join(page.extract_text() for page in reader.pages)
# Squash whitespace AND hyphens: pdf extraction scatters whitespace, and
# LaTeX hyphenates across line breaks ("cen-sored"), which defeated the
# first version of this check -- a verifier artifact, not a document one.
squashed = re.sub(r"[\s\-]+", "", text)

print(f"pages: {len(reader.pages)}   text layer: {len(text):,} chars\n")

failures = []


def must_contain(label, *needles, where=squashed):
    ok = all(n in where for n in needles)
    print(f"  [{'OK' if ok else 'MISSING'}] {label}")
    if not ok:
        missing = [n for n in needles if n not in where]
        print(f"            missing: {missing}")
        failures.append(label)


def must_not_contain(label, needle, allowed_context=None):
    """Fail if needle appears OUTSIDE the allowed withdrawal context."""
    idxs = [m.start() for m in re.finditer(re.escape(needle), squashed)]
    bad = []
    for i in idxs:
        ctx = squashed[max(0, i - 400):i + 200]
        if allowed_context and allowed_context in ctx:
            continue
        bad.append(ctx[:100])
    ok = not bad
    print(f"  [{'OK' if ok else 'FOUND'}] retired outside withdrawal: {label}"
          f" ({len(idxs)} occurrence(s), {len(idxs)-len(bad)} in withdrawal text)")
    if bad:
        print(f"            stray context: ...{bad[0]}...")
        failures.append(label)


print("1. Corrected numbers present in the compiled output:")
must_contain("Table III t50 127/128", "127±5", "128±5")
must_contain("Table III decay pair", "0.261±0.017", "0.0499")
must_contain("censoring 50/50", "(50/50c.)")  # table abbreviates; caption expands "c."
must_contain("energy ratio 5.2x", "5.2×")
must_contain("oracle margin 1.55x", "1.55×")
must_contain("oracle sensitivity row", "199±5", "1454±137")
must_contain("proxy correction +1.81", "+1.81")
must_contain("mean shift 0.0159", "0.0159±0.0013")

print("\n2. Retired figures absent outside their withdrawal narratives:")
must_not_contain("1999", "1999")
must_not_contain("527 ticks", "527ticks")
must_not_contain("+0.04 proxy", "+0.04",
                 allowed_context="Apreviousrevisionofthiswork")
must_not_contain("4.5x ratio", "4.5×", allowed_context="correction")
must_not_contain("15 million", "15million")
must_not_contain("perfectly matches", "perfectlymatches")
must_not_contain("thermodynamic braking", "thermodynamicbraking")

print("\n3. Placeholders and removed-author material:")
# Author decision (2026-08-18): the 4th author was removed, and the
# placeholder ORCID belonged to her -- BOTH must now be ABSENT.
must_not_contain("removed ORCID", "0000000198765432")
must_not_contain("removed author Shruthi", "Shruthi")
# The IEEEtran funding-agency thanks remains the one visible template
# placeholder, left for the author to fill or delete before submission.
must_contain("funding-agency placeholder", "Identifyapplicablefundingagency")

print("\n4. Author block (3 authors, per author decision 2026-08-18):")
for name in ("ShivamMalge", "PrajwalNarendraHegde", "KoushikKR"):
    must_contain(f"author: {name}", name)

print("\n5. Structure:")
must_contain("provenance subsection",
             "ProvenanceandStabilityoftheReportedFigures")
must_contain("eviction subsection", "BeliefLifetimeandEviction")
n_figs = sum(1 for p in reader.pages for _ in (p.images or []))
print(f"  [{'OK' if n_figs >= 5 else 'LOW'}] embedded images: {n_figs} (expect >= 5)")
if n_figs < 5:
    failures.append("figure count")

print("\n" + ("PDF VERIFICATION FAILED: " + ", ".join(failures)
              if failures else "PDF VERIFICATION CLEAN."))
sys.exit(1 if failures else 0)
