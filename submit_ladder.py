#!/usr/bin/env python
"""Fire the constrained-search candidate ladder at the live grader, stop on accept."""
import time
from ssh_submit import attempt

INNER = "Hup-la... He-ha... He-ho... Z-z-z-z..."
EPSON = "[EPSON W-02]"

CANDS = [
    # wrapper forms, highest model-confidence first
    ("1  exact (newlines)",      f"flag{{{INNER}\n\n{EPSON}}}"),
    ("2  newlines->space",       f"flag{{{INNER} {EPSON}}}"),
    ("3  single-line, no EPSON", f"flag{{{INNER}}}"),
    # inner-only (in case the grader adds the flag{} wrapper itself)
    ("4  inner exact (newlines)",f"{INNER}\n\n{EPSON}"),
    ("5  inner newlines->space", f"{INNER} {EPSON}"),
    ("6  inner single-line",     INNER),
]

POS = ("correct", "parab", "congrat", "accept", "first blood", "winner", "you win", "well done", "nice")

for label, flag in CANDS:
    try:
        ok, out = attempt(flag, verbose=False)
    except Exception as e:
        print(f"\n##### {label}\n  ERROR: {e!r}")
        time.sleep(1.0); continue
    low = out.lower()
    pos_hit = [w for w in POS if w in low]
    print("\n" + "#"*70)
    print(f"### {label}")
    print(f"### sent: {flag!r}")
    print(f"### heuristic_accepted={ok}  positive_words={pos_hit}")
    print("--- rendered screen ---")
    print(out)
    print("--- end screen ---")
    if ok and pos_hit:
        print(f"\n*** LIKELY ACCEPTED on {label} ***")
        break
    time.sleep(1.0)
print("\nladder complete.")
