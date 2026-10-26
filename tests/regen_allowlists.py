"""Regenerate the purity allowlists from what the tree actually contains.

The gate's rule is that its budget table may only shrink, and every shrink has to
land in the same change that caused it -- otherwise the numbers drift away from
reality and the gate stops meaning anything. After a migration step that removes
offenders, the tree reports ``LOOSE`` (budget higher than reality) and ``STALE``
(entry whose file no longer offends) for dozens of files, and lowering them by
hand is where transcription mistakes come from.

So this tool reuses the gate's own scanner: it refuses to run while the tree has a
``NEW`` violation (an offender with no budget at all -- a real regression, which
regenerating would silently bless), and otherwise rewrites the literal to exactly
what the scanner sees.

``LOOSE`` and ``STALE`` are not regressions; they are the gate asking for its
budget to be lowered, which is precisely what this does.

Usage::

    python tests/regen_allowlists.py            # dry run, print the new table
    python tests/regen_allowlists.py --write    # rewrite the literal in place

Module: ``coffin/tests``. A maintenance tool, not a test: ``unittest discover``
matches ``test_*.py`` only, so this file is not collected.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_backends_purity as gate  # noqa: E402

#: The marker the gate uses for "violation with no budget entry". Anything else
#: (``OVER``, ``LOOSE``, ``STALE``) is a budget that disagrees with the tree, which
#: is what regenerating is for.
REGRESSION_MARKER = "NEW"

#: The literal this tool rewrites.
_ANCHOR = "ALLOWLISTS: dict[str, dict[str, dict[str, int]]] = {"


def regressions(scan: gate.Scan) -> list[str]:
    """Collect the violations that no budget covers.

    Args:
        scan: A scan audited against the current allowlists.

    Returns:
        One line per ``NEW`` violation. Non-empty means the tree has an offender
        that was introduced rather than budgeted, so the table must not be
        regenerated over it.
    """
    out = []
    for name, report in scan.reports.items():
        for line in report.failure_lines():
            if line.lstrip().startswith(REGRESSION_MARKER):
                out.append(f"{name}: {line.strip()}")
    return out


def main(argv: list[str] | None = None) -> int:
    """Print or write the allowlist that matches the tree as it is now.

    Args:
        argv: Command line arguments; ``--write`` rewrites the file.

    Returns:
        Process exit code: 1 when the tree has a regression that would be blessed
        by regenerating, else 0.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true",
                        help="rewrite the literal in place instead of printing it")
    args = parser.parse_args(argv)

    current = gate.audit()
    bad = regressions(current)
    if bad:
        print("refusing to regenerate: the tree has violations with no budget entry.")
        print("These are introduced offenders, not a budget that needs lowering;"
              " regenerating would whitelist them.")
        for line in bad:
            print("  " + line)
        return 1

    # An empty budget per rule makes every occurrence visible, which is what
    # ``format_allowlist`` needs to build the table from scratch.
    fresh = gate.audit(budgets={rule.name: {} for rule in gate.RULES})
    body = gate.format_allowlist(fresh)

    if not args.write:
        print(body)
        return 0

    path = Path(__file__).resolve().parent / "test_backends_purity.py"
    source = path.read_text(encoding="utf-8")
    start = source.index(_ANCHOR)
    end = source.index("\n}\n", start) + len("\n}")
    path.write_text(source[:start] + body.rstrip("\n") + source[end:], encoding="utf-8")
    print(f"rewrote {path.name}")
    for report in fresh.reports.values():
        print("  " + report.totals_line())
    return 0


if __name__ == "__main__":
    sys.exit(main())
