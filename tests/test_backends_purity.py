"""Purity gate: hardware detail must not leak out of ``coffin/backends``.

Module: ``coffin/tests``. Standard library only, so the gate runs on a CPU-only
host with no DCU and no compiled extension, and costs nothing at import time.

The one-shot ``hip_ops`` -> backend refactor touches roughly 150 call sites. This
module is the safety net around that rename: it walks the real ``coffin`` tree on
every run and fails when a file outside the backend package reaches into
hardware-specific detail.

Constraints:
    * Every run rescans the filesystem (``pathlib.Path.rglob``); a fixed list of
      known offenders would go stale the moment a file is added, which is
      exactly when the gate has to fire.
    * Hits are grouped (``prod`` vs ``probe``) and each group carries its own
      allowlist, so progress on production code cannot be masked by ``tests/``,
      which is allowed to stay platform-specific longest. One-off probes live
      outside this repository (in the parent tree's ``diag/``), so the gate never
      sees them and a new probe never needs an allowlist entry.
    * A group's allowlist is per file and holds the exact hit count of that file
      today. More hits fail (a regression), fewer hits fail (the entry must be
      lowered in the same change), and an entry whose file no longer offends
      fails (delete it). That is what makes every allowlist converge on empty
      instead of becoming a permanent exemption list.
    * Comments and docstrings count as offenders. A hardware fact stated in a
      comment outside the backend is the same coupling in prose: it tells the
      next reader to edit the platform-specific file, so it moves with the
      backend rather than staying behind.
    * Counts are offending lines, and a line counts once per rule even when it
      matches several patterns of that rule, so "hits" is a progress metric
      rather than a regex match count.
    * The scanner is exempt from its own rules: it necessarily spells out the
      literals it searches for. Every rule is proven live by a synthetic bait
      tree, so a green run still means clean code.
    * The bait tree lives in ``coffin/.tmp_purity`` (gitignored), created with a
      plain ``Path.mkdir()`` and removed with ``shutil.rmtree(ignore_errors=True)``.
      Do not switch it to ``tempfile``: on Windows a directory created with mode
      ``0o700`` (which ``mkdtemp`` uses) gets a DACL the sandboxed token cannot
      delete, and the leftovers then break ``git status`` for everyone.
"""

from __future__ import annotations

import re
import shutil
import sys
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]

#: Posix path of this file, relative to ``coffin``. Exempted from every rule as
#: explained in the module docstring.
SELF = "tests/test_backends_purity.py"

#: Scratch root for the synthetic bait tree. Inside the workspace and gitignored
#: so a crashed run cannot pollute ``git status``.
BAIT_ROOT = ROOT / ".tmp_purity"

SOURCE_SUFFIXES = frozenset({".py", ".cu", ".c", ".h", ".cpp", ".cc", ".hpp"})

SKIP_DIRECTORIES = frozenset({
    "__pycache__", ".git", ".hg", ".svn", ".venv", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", ".tox", "node_modules",
    ".tmp_purity",
})


@dataclass(frozen=True)
class Pattern:
    """One regex that recognises a hardware-specific token.

    Attributes:
        label: Short name used in reports to split a rule's hits by flavour.
        regex: Compiled expression; a match anywhere on a line is an offender.
    """

    label: str
    regex: re.Pattern


@dataclass(frozen=True)
class Group:
    """A slice of the tree with its own budget, so progress is readable.

    Attributes:
        name: ``"prod"`` for code that ships, ``"probe"`` for ``tests/``,
            which may stay platform-specific longest.
        prefixes: Directory prefixes in this group. An empty tuple makes the
            group the catch-all and it must be declared last.
    """

    name: str
    prefixes: tuple[str, ...]


PROD = Group("prod", ())
PROBE = Group("probe", ("tests/",))
SHIPPED = (PROD,)
WITH_PROBES = (PROBE, PROD)


@dataclass(frozen=True)
class Rule:
    """One purity rule, its scan scope, its groups, and the allowlist to shrink.

    Attributes:
        name: Stable rule id, used as the unittest method suffix and in reports.
        summary: One sentence stating what the rule protects and why.
        patterns: Recognisers; a line matching any of them is one offender line.
        scope: Directory prefixes to scan. Empty means the whole tree.
        exempt: Directory prefixes whose hits are reported but never budgeted --
            the backend package itself. The scanner
            file is always exempt as well.
        groups: Budget slices, probe first then the ``prod`` catch-all.
        budgets: Allowlist as ``group -> posix relative path -> exact lines``.
    """

    name: str
    summary: str
    patterns: tuple[Pattern, ...]
    groups: tuple[Group, ...] = SHIPPED
    scope: tuple[str, ...] = ()
    exempt: tuple[str, ...] = ()
    budgets: dict[str, dict[str, int]] = field(default_factory=dict)


@dataclass(frozen=True)
class Occurrence:
    """A single offending line, addressed the way an editor jumps to it."""

    path: str
    line: int
    label: str
    text: str

    def render(self) -> str:
        """Format as ``path:line: text`` so a failure list is directly actionable.

        Returns:
            One report line, truncated at 200 characters of source text.
        """
        return f"{self.path}:{self.line}: {self.text[:200]}"


@dataclass(frozen=True)
class GroupReport:
    """The scan result for one group of one rule.

    Attributes:
        group: The group this slice covers.
        allowed: The allowlist in force for this group, ``path -> lines``.
        per_file: Tracked hit count per file.
        violations: Entries that are new, over budget, or under budget, as
            ``(path, actual, allowed)`` with ``allowed = None`` for a new file.
        stale: Budget keys that no longer offend; they must be deleted.
    """

    group: Group
    allowed: dict[str, int]
    per_file: dict[str, int]
    violations: tuple[tuple[str, int, int | None], ...]
    stale: tuple[str, ...]

    @property
    def budget(self) -> int:
        """Total offending lines this group's allowlist still tolerates."""
        return sum(self.allowed.values())

    @property
    def tracked(self) -> int:
        """Offending lines currently found in this group."""
        return sum(self.per_file.values())

    @property
    def ok(self) -> bool:
        """True when this group has no violation and no stale entry."""
        return not self.violations and not self.stale

    def totals_line(self) -> str:
        """Format this slice's progress, which is the migration metric.

        Returns:
            ``"<group>: tracked=<n> budget=<n> files=<n>"``.
        """
        return (f"{self.group.name}: tracked={self.tracked}"
                f" budget={self.budget} files={len(self.per_file)}")


@dataclass(frozen=True)
class RuleReport:
    """The scan result for one rule across all its groups.

    Attributes:
        rule: The rule that produced this report.
        scanned: Number of source files visited under the scan root.
        hits: All occurrences in scope, exempt ones included.
        exempt: Occurrences under the rule's exempt roots, or in the scanner.
        groups: Per-group results, in rule declaration order.
    """

    rule: Rule
    scanned: int
    hits: tuple[Occurrence, ...]
    exempt: tuple[Occurrence, ...]
    groups: tuple[GroupReport, ...]

    @property
    def tracked(self) -> int:
        """Offending lines in every group."""
        return sum(group.tracked for group in self.groups)

    @property
    def budget(self) -> int:
        """Offending lines the whole allowlist still tolerates."""
        return sum(group.budget for group in self.groups)

    @property
    def ok(self) -> bool:
        """True when every group passes."""
        return all(group.ok for group in self.groups)

    def totals_line(self) -> str:
        """Format the progress line printed for this rule.

        Returns:
            The rule name, then each group's slice, then the exempt count.
        """
        slices = "  ".join(f"[{group.totals_line()}]" for group in self.groups)
        return (f"{self.rule.name}: {slices}  hits={len(self.hits)}"
                f" exempt={len(self.exempt)}")

    def failure_lines(self) -> list[str]:
        """Explain every violation and stale entry as actionable lines.

        Returns:
            Human-readable lines; empty when the rule passes. Each offending file
            gets its reason followed by its ``path:line: text`` occurrences.
        """
        lines: list[str] = []
        for group in self.groups:
            for path, actual, allowed in group.violations:
                where = f"{path}" if group.group.name == "prod" else f"{path} [{group.group.name}]"
                if allowed is None:
                    lines.append(f"  NEW {where}: {actual} offending line(s), no allowlist entry")
                elif actual > allowed:
                    lines.append(f"  OVER {where}: {actual} offending line(s) > allowlist {allowed}")
                else:
                    lines.append(f"  LOOSE {where}: allowlist {allowed} > {actual} offending"
                                 f" line(s); lower it to {actual}")
                lines.extend(f"    {occurrence.render()}" for occurrence in self.hits
                             if occurrence.path == path
                             and not (occurrence.path == SELF
                                      or _under(occurrence.path, self.rule.exempt)))
            for path in group.stale:
                lines.append(f"  STALE allowlist entry, delete it: {path} [{group.group.name}]")
        return lines


@dataclass(frozen=True)
class Scan:
    """Every rule's report for one filesystem root.

    Attributes:
        root: Root the scan walked.
        scanned: Number of source files visited.
        reports: Per-rule results keyed by rule name, in rule declaration order.
    """

    root: Path
    scanned: int
    reports: dict[str, RuleReport]

    @property
    def ok(self) -> bool:
        """True when every rule passes."""
        return all(report.ok for report in self.reports.values())


def _under(path: str, prefixes: tuple[str, ...]) -> bool:
    """True if a posix relative path is inside one of the directory prefixes.

    Args:
        path: Posix relative path of a source file.
        prefixes: Directory prefixes, with or without a trailing slash.

    Returns:
        ``True`` when ``path`` equals a prefix or starts with it plus ``/``.
    """
    for prefix in prefixes:
        trimmed = prefix.rstrip("/")
        if path == trimmed or path.startswith(trimmed + "/"):
            return True
    return False


def _group_of(rule: Rule, path: str) -> Group:
    """Pick the group a tracked path belongs to.

    Args:
        rule: Rule whose groups apply.
        path: Posix relative path.

    Returns:
        The first group whose prefixes contain the path; the last group must be
        the catch-all, so this never falls through for a well-formed rule.
    """
    for group in rule.groups:
        if group.prefixes and _under(path, group.prefixes):
            return group
    return rule.groups[-1]


def _iter_sources(root: Path) -> Iterator[tuple[Path, str]]:
    """Walk the tree and yield every source file with its posix relative path.

    The walk is the point of the gate: it must see files the author of this test
    has never heard of, so there is no suffix or directory list to keep in sync
    with the tree.

    Args:
        root: Directory to walk.

    Yields:
        ``(absolute path, posix relative path)`` for each ``.py`` / CUDA / C /
        C++ source file, sorted by path for a stable failure order.
    """
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        relative = path.relative_to(root)
        if SKIP_DIRECTORIES.intersection(relative.parts[:-1]):
            continue
        yield path, relative.as_posix()


def _scan_rule(rule: Rule, root: Path) -> tuple[Occurrence, ...]:
    """Collect every offending line for one rule under ``root``.

    Args:
        rule: Rule whose patterns and scope apply.
        root: Scan root.

    Returns:
        Offending lines in path then line order, at most one per line per rule.
    """
    occurrences: list[Occurrence] = []
    for path, relative in _iter_sources(root):
        if rule.scope and not _under(relative, rule.scope):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), 1):
            for pattern in rule.patterns:
                if pattern.regex.search(line):
                    occurrences.append(
                        Occurrence(relative, number, pattern.label, line.strip()))
                    break
    return tuple(occurrences)


def audit(root: Path = ROOT, budgets: dict[str, dict[str, dict[str, int]]] | None = None) -> Scan:
    """Run every purity rule against a tree.

    Args:
        root: Tree to scan; ``coffin`` by default.
        budgets: Optional allowlist override as ``rule -> group -> path -> lines``.
            ``None`` uses the declared allowlists; an empty mapping audits a tree
            as if no allowlist existed, which is how the allowlists were
            generated.

    Returns:
        A :class:`Scan` with one report per rule.
    """
    root = Path(root)
    scanned = sum(1 for _ in _iter_sources(root))
    reports: dict[str, RuleReport] = {}
    for rule in RULES:
        effective = rule.budgets if budgets is None else budgets.get(rule.name, {})
        hits = _scan_rule(rule, root)
        exempt = tuple(o for o in hits
                       if o.path == SELF or _under(o.path, rule.exempt))
        tracked = tuple(o for o in hits
                        if o.path != SELF and not _under(o.path, rule.exempt))

        group_reports: list[GroupReport] = []
        for group in rule.groups:
            allowed = effective.get(group.name, {})
            per_file: dict[str, int] = {}
            for occurrence in tracked:
                if _group_of(rule, occurrence.path) is group:
                    per_file[occurrence.path] = per_file.get(occurrence.path, 0) + 1
            violations = tuple(
                (path, actual, allowed.get(path))
                for path, actual in sorted(per_file.items())
                if allowed.get(path) != actual
            )
            stale = tuple(sorted(path for path in allowed if path not in per_file))
            group_reports.append(
                GroupReport(group=group, allowed=allowed, per_file=per_file,
                            violations=violations, stale=stale))
        reports[rule.name] = RuleReport(
            rule=rule, scanned=scanned, hits=hits, exempt=exempt,
            groups=tuple(group_reports),
        )
    return Scan(root=root, scanned=scanned, reports=reports)


# Allowlists. Each entry is the exact number of offending lines that file
# contains today, per group. Regenerate after a migration step with
# ``python tests/test_backends_purity.py --emit-allowlist``; the gate fails on
# NEW, OVER, LOOSE and STALE entries, so these tables can only shrink.
ALLOWLISTS: dict[str, dict[str, dict[str, int]]] = {
    "R1_hip_ops_import": {
        "probe": {
            "tests/support.py": 1,
            "tests/test_attention.py": 1,
            "tests/test_ops.py": 7,
        },
        "prod": {
        },
    },
    "R2_platform_literal": {
        "probe": {
        },
        "prod": {
        },
    },
    "R3_cuda_literal": {
        "probe": {
            "tests/support.py": 7,
            "tests/test_attention.py": 3,
            "tests/test_backends.py": 4,
            "tests/test_mtp.py": 4,
            "tests/test_openai_mm.py": 2,
            "tests/test_ops.py": 37,
            "tests/test_vision.py": 18,
        },
        "prod": {
        },
    },
    "R4_kernels_path": {
        "probe": {
            "tests/test_attention.py": 1,
        },
        "prod": {
        },
    },
    "R5_backend_dependency": {
        "prod": {
        },
    },
}


RULES: tuple[Rule, ...] = (
    Rule(
        name="R1_hip_ops_import",
        summary="Only the backend package may import the top-level ``hip_ops`` "
                "module; every other layer goes through the op table.",
        patterns=(
            Pattern("import_hip_ops",
                    re.compile(r"^\s*(?:import\s+hip_ops\b|from\s+hip_ops\s+import\b)")),
            # The import statement is not the only way in: a dynamic import and a
            # ``sys.modules`` poke both reach the shim without a matching line,
            # and those are exactly the forms a "temporary" bypass would use.
            Pattern("dynamic_hip_ops",
                    re.compile(r"(?:import_module|sys\.modules)\s*[\(\[]\s*[\"']hip_ops[\"']")),
        ),
        groups=WITH_PROBES,
        exempt=("backends/",),
        budgets=ALLOWLISTS["R1_hip_ops_import"],
    ),
    Rule(
        name="R2_platform_literal",
        summary="gfx906, --offload-arch, PYTORCH_ROCM_ARCH and "
                "TORCH_EXTENSIONS_DIR are build facts of one accelerator and "
                "belong to the backend that builds for it.",
        patterns=(
            Pattern("gfx906", re.compile(r"gfx906")),
            Pattern("offload_arch", re.compile(r"--offload-arch")),
            Pattern("rocm_arch_env", re.compile(r"PYTORCH_ROCM_ARCH")),
            Pattern("extensions_dir_env", re.compile(r"TORCH_EXTENSIONS_DIR")),
        ),
        groups=WITH_PROBES,
        exempt=("backends/",),
        budgets=ALLOWLISTS["R2_platform_literal"],
    ),
    Rule(
        name="R3_cuda_literal",
        summary="The torch CUDA spellings -- device strings, .cuda() and the "
                "torch.cuda runtime API -- name one vendor's stack; a non-CUDA "
                "accelerator has its own device string and its own timing, "
                "capture and memory calls.",
        patterns=(
            Pattern("device_literal", re.compile(r"""["']cuda(?::\d+)?["']""")),
            Pattern("cuda_method", re.compile(r"\.cuda\(\)")),
            Pattern("torch_cuda_api", re.compile(r"\btorch\.cuda\b")),
            Pattern("cuda_api_bare", re.compile(
                r"(?<![\w.])cuda\.(?:Event|Stream|CUDAGraph|graph|graph_pool_handle"
                r"|synchronize|is_available|empty_cache|mem_get_info|memory_allocated"
                r"|memory_reserved|max_memory_allocated|reset_peak_memory_stats"
                r"|current_stream|set_device|get_device|device_count"
                r"|get_device_properties|get_device_name|ipc_collect|init)\b")),
        ),
        groups=WITH_PROBES,
        exempt=("backends/",),
        budgets=ALLOWLISTS["R3_cuda_literal"],
    ),
    Rule(
        name="R4_kernels_path",
        summary="Kernel sources are a backend asset; a path into ``kernels/`` "
                "outside the backend couples the caller to one build layout.",
        patterns=(
            Pattern("kernel_source_path", re.compile(
                r"(?<![\w.])kernels[/\\][\w./\\-]*?\.(?:cu|cpp|cc|c|h|hpp)\b")),
            Pattern("kernels_dir_path",
                    re.compile(r"(?<![\w.])kernels[/\\](?:``|`|\"|'|\s|$)")),
            Pattern("kernels_dir_name", re.compile(r"""["']kernels["']""")),
        ),
        groups=WITH_PROBES,
        exempt=("backends/",),
        budgets=ALLOWLISTS["R4_kernels_path"],
    ),
    Rule(
        name="R5_backend_dependency",
        summary="The dependency arrow points down only: a backend may import the "
                "standard library, torch and its own package, never the model, "
                "ops, engine, serving or cache layers.",
        patterns=(
            Pattern("layer_import", re.compile(
                r"^\s*(?:from|import)\s+(?:coffin\.)?"
                r"(?:ops|models|engine|serving|cache)(?:\.[\w.]*)?\s*"
                r"(?:import\b|as\b|,|$)")),
            # A relative import is the form a backend author actually reaches for
            # ("from ..ops import x"), and it needs its own pattern: the absolute
            # one above cannot match the leading dots.
            Pattern("relative_layer_import", re.compile(
                r"^\s*from\s+\.{1,3}(?:ops|models|engine|serving|cache)\b")),
        ),
        groups=SHIPPED,
        scope=("backends/",),
        budgets=ALLOWLISTS["R5_backend_dependency"],
    ),
)


def _write_tree(root: Path, files: dict[str, str]) -> None:
    """Materialise a synthetic source tree for the bait cases.

    Args:
        root: Directory to create the files under.
        files: Mapping of relative posix path to file body.
    """
    for name, body in files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")


BAIT = {
    "bait_r1.py": "import hip_ops\n",
    # The bypasses an import-statement-only rule would miss: a dynamic import and
    # a ``sys.modules`` poke both reach the shim with no matching import line.
    "bait_r1_dyn.py": ('import importlib\n'
                       'SHIM = importlib.import_module("hip_ops")\n'),
    "bait_r1_modules.py": 'import sys\nCACHED = sys.modules["hip_ops"]\n',
    "bait_r2.py": 'ARCH = "gfx906"\nFLAGS = ["--offload-arch=gfx906"]\n',
    "bait_r3.py": ('x = torch.zeros(1, device="cuda")\n'
                   'y = x.cuda()\ntorch.cuda.synchronize()\n'),
    "bait_r4.py": 'HERE = "kernels/gemv.cu"\nRATE = "12 kernels/step"\n',
    # Rule 5 only scans ``backends/``, so its bait has to live there. The relative
    # form is the one an author writing inside the package would reach for.
    "backends/bait_r5.py": "from ops.dtypes import D\nimport models.qwen35\n",
    "backends/bait_r5_rel.py": "from ..ops.dtypes import D\n",
    "__pycache__/junk.py": "torch.cuda.synchronize()\n",
    ".tmp_purity/junk.py": "torch.cuda.synchronize()\n",
}


class PurityGateTest(unittest.TestCase):
    """Every rule passes against the current tree and every allowlist is current."""

    scan: Scan

    @classmethod
    def setUpClass(cls) -> None:
        """Scan once per run and print the per-rule totals.

        The ``prod`` slice is the number that must reach zero; the totals line is
        the migration progress metric for each rule.
        """
        cls.scan = audit()
        print(f"\n[platform purity] root={cls.scan.root} source_files={cls.scan.scanned}")
        for report in cls.scan.reports.values():
            print(f"[platform purity] {report.totals_line()}")

    def _assert_rule(self, name: str) -> None:
        """Fail with actionable lines when a rule has violations or stale entries.

        Args:
            name: Rule id from :data:`RULES`.
        """
        report = self.scan.reports[name]
        if report.ok:
            return
        problem = [f"{name}: {report.rule.summary}",
                   f"  ({report.totals_line()})"]
        problem.extend(report.failure_lines())
        self.fail("\n".join(problem))

    def test_rule1_no_hip_ops_import_outside_backend(self):
        """Only ``backends/`` may import ``hip_ops``."""
        self._assert_rule("R1_hip_ops_import")

    def test_rule2_no_platform_build_literals_outside_backend(self):
        """Architecture strings and extension build knobs live in the backend only."""
        self._assert_rule("R2_platform_literal")

    def test_rule3_no_cuda_spelling_outside_backend(self):
        """Device strings, ``.cuda()`` and ``torch.cuda`` are backend detail."""
        self._assert_rule("R3_cuda_literal")

    def test_rule4_no_kernels_path_outside_backend(self):
        """No caller outside the backend names a file under ``kernels/``."""
        self._assert_rule("R4_kernels_path")

    def test_rule5_backends_do_not_import_upward(self):
        """The backend package must not import the layers that depend on it."""
        self._assert_rule("R5_backend_dependency")

    def test_scan_reaches_the_known_tree(self):
        """The walk must see the real tree, not an empty or partial one.

        Guards against a scan root that silently resolves to the wrong directory:
        the gate would then pass without looking at anything.

        The assertion is about files the *walk* visited, not about files that have
        hits. Keying it on a hit pinned both the layout and the migration's own
        progress: ``hip_ops.py`` stopped existing when the kernels moved into the
        backend, and ``models/qwen35/model.py`` stopped offending once its device
        literals were replaced -- a cleaner tree must not fail this test.
        """
        self.assertGreater(self.scan.scanned, 100)
        walked = {path.relative_to(self.scan.root).as_posix()
                  for path, _ in _iter_sources(self.scan.root)}
        for expected in ("models/qwen35/model.py", "tests/support.py",
                         "backends/reference/__init__.py", "ops/gemv.py"):
            self.assertIn(expected, walked)

    def test_production_is_separated_from_probe_directories(self):
        """Rule 3 must report shipping code apart from ``tests/``.

        A test-only allowance would otherwise hide a regressed production file,
        and the production number is the one the migration had to drive to zero.

        Production is now at zero, so the assertion is that the split still exists
        and that nothing in the ``prod`` group lives under a probe directory. It
        used to also require ``prod`` to be non-empty; that pinned the migration's
        own progress and would fail on the very state it was driving toward.
        """
        report = self.scan.reports["R3_cuda_literal"]
        groups = {group.group.name: group for group in report.groups}
        self.assertEqual(set(groups), {"prod", "probe"})
        self.assertTrue(all(not path.startswith("tests/")
                            for path in groups["prod"].per_file))
        self.assertTrue(all(path.startswith("tests/")
                            for path in groups["probe"].per_file))
        # The v3 rule's namesake property: no shipped file may still spell a
        # vendor's device API. Probes may, and do.
        self.assertEqual(groups["prod"].tracked, 0,
                         "shipped code must not name one vendor's device stack")

    def test_rules_catch_synthetic_offenders(self):
        """Every rule fires on a bait tree, so a green run means clean code.

        The bait also pins the noise handling: a violating file under
        ``__pycache__`` or under the gate's own scratch root must be ignored, and
        ``12 kernels/step`` (a rate, not a path) must not count as a kernel
        reference. The budget outcomes pin the four verdicts the gate can return.
        """
        BAIT_ROOT.mkdir(parents=True, exist_ok=True)
        self.addCleanup(shutil.rmtree, BAIT_ROOT, ignore_errors=True)
        root = BAIT_ROOT / "case"
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        _write_tree(root, BAIT)

        empty = {rule.name: {} for rule in RULES}
        scan = audit(root, budgets=empty)
        caught = {name: {path for group in report.groups
                         for path, _actual, _allowed in group.violations}
                  for name, report in scan.reports.items()}
        self.assertEqual(caught["R1_hip_ops_import"],
                         {"bait_r1.py", "bait_r1_dyn.py", "bait_r1_modules.py"})
        self.assertEqual(caught["R2_platform_literal"], {"bait_r2.py"})
        self.assertEqual(caught["R3_cuda_literal"], {"bait_r3.py"})
        self.assertEqual(caught["R4_kernels_path"], {"bait_r4.py"})
        self.assertEqual(caught["R5_backend_dependency"],
                         {"backends/bait_r5.py", "backends/bait_r5_rel.py"})
        self.assertEqual(self._per_file(scan, "R3_cuda_literal"), {"bait_r3.py": 3})
        self.assertEqual(self._per_file(scan, "R4_kernels_path"), {"bait_r4.py": 1})

        def with_budget(name: str, count: int) -> Scan:
            return audit(root, budgets={name: {"prod": {"bait_r3.py": count}}})

        self.assertTrue(with_budget("R3_cuda_literal", 3)
                        .reports["R3_cuda_literal"].ok)
        self.assertEqual(self._violations(with_budget("R3_cuda_literal", 2),
                                          "R3_cuda_literal"),
                         (("bait_r3.py", 3, 2),))
        self.assertEqual(self._violations(with_budget("R3_cuda_literal", 9),
                                          "R3_cuda_literal"),
                         (("bait_r3.py", 3, 9),))
        stale = audit(root, budgets={"R3_cuda_literal": {"prod": {"gone.py": 1}}})
        self.assertEqual(
            tuple(path for group in stale.reports["R3_cuda_literal"].groups
                  for path in group.stale),
            ("gone.py",))

    @staticmethod
    def _per_file(scan: Scan, name: str) -> dict[str, int]:
        """Merge a rule's per-group hit counts.

        Args:
            scan: Scan result.
            name: Rule id.

        Returns:
            ``path -> offending lines`` across every group.
        """
        merged: dict[str, int] = {}
        for group in scan.reports[name].groups:
            merged.update(group.per_file)
        return merged

    @staticmethod
    def _violations(scan: Scan, name: str) -> tuple[tuple[str, int, int | None], ...]:
        """Collect a rule's violations across every group.

        Args:
            scan: Scan result.
            name: Rule id.

        Returns:
            Tuples of ``(path, actual, allowed)``.
        """
        return tuple(violation for group in scan.reports[name].groups
                     for violation in group.violations)


def format_allowlist(scan: Scan) -> str:
    """Render the current per-file hit counts as an allowlist literal.

    Args:
        scan: A scan audited with empty budgets.

    Returns:
        Python source for :data:`ALLOWLISTS`, sorted, ready to paste.
    """
    blocks = []
    for name, report in scan.reports.items():
        group_blocks = []
        for group in report.groups:
            entries = "".join(f'            "{path}": {count},\n'
                              for path, count in sorted(group.per_file.items()))
            group_blocks.append(f'        "{group.group.name}": {{\n{entries}        }},\n')
        blocks.append(f'    "{name}": {{\n' + "".join(group_blocks) + "    },\n")
    return ("ALLOWLISTS: dict[str, dict[str, dict[str, int]]] = {\n"
            + "".join(blocks) + "}\n")


def main(argv: list[str]) -> int:
    """Print the audit, or the allowlist that matches the tree as it is now.

    ``--emit-allowlist`` refuses to run while the tree is red. Regenerating the
    budgets from a red tree would bake the current offenders in as the new
    baseline, which is the one way an allowlist can grow; requiring green first
    turns that into an error the author has to fix rather than a silent accept.

    Args:
        argv: Command line arguments; ``--report`` or ``--emit-allowlist``.

    Returns:
        Process exit code: 1 when the tree breaks a rule (or when
        ``--emit-allowlist`` is asked for on a red tree), else 0.
    """
    if "--emit-allowlist" in argv:
        current = audit()
        if not current.ok:
            print("refusing to emit an allowlist: the tree already breaks a rule.")
            print("Fix the offenders (or lower the budget for the file that improved)"
                  " first; regenerating now would whitelist them.")
            for report in current.reports.values():
                if report.ok:
                    continue
                print(f"\n{report.rule.name}: {report.rule.summary}")
                print("\n".join(report.failure_lines()))
            return 1
        print(format_allowlist(audit(budgets={rule.name: {} for rule in RULES})))
        return 0
    scan = audit()
    print(f"root={scan.root}")
    print(f"source_files={scan.scanned}")
    for report in scan.reports.values():
        print(report.totals_line())
    for report in scan.reports.values():
        if report.ok:
            continue
        print(f"\n{report.rule.name}: {report.rule.summary}")
        print("\n".join(report.failure_lines()))
    return 0 if scan.ok else 1


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(main(sys.argv[1:]))
    unittest.main()