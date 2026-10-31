"""纯度门禁：硬件细节不许漏出 ``coffin/backends``。

Module: ``coffin/tests``。只用标准库，所以在没有 DCU、没有编译扩展的纯 CPU
主机上也能跑，import 阶段不花代价。

``hip_ops`` -> 后端的一次性重构要动约 150 个调用点。本模块是那次改名周围的
安全网：它每轮都走真实的 ``coffin`` 目录树，后端包之外的文件只要伸手去够硬件
相关细节就失败。

Constraints:
    * 每次都重扫文件系统（``pathlib.Path.rglob``）；写死一份已知违规清单，
      文件一新增就过期，而新增文件正是门禁必须触发的时刻。
    * 命中按 ``prod`` / ``probe`` 分组，每组带自己的允许列表，``tests/`` 允许最晚
      平台化，却不会因此掩盖生产代码的进度。一次性探针放在本仓库之外（父级目录
      树的 ``diag/``），门禁看不到它们，新增探针也不必登记条目。
    * 一个组的允许列表按文件记录该文件今天的精确命中行数。多一行失败（回归），
      少一行失败（必须在同一次改动里下调），条目对应的文件不再违规也失败
      （删掉它）。每份允许列表因此只能朝空收敛，不会变成永久豁免表。
    * 注释和 docstring 也算违规。后端之外用注释写下的硬件事实是同一种耦合的
      散文形态：它在提示下一个读者去改平台相关文件，所以它该随后端一起走。
    * 计数单位是违规行；一行即使命中同一条规则的多个模式也只算一次，所以
      「命中数」是进度指标，不是正则匹配次数。
    * 扫描器不受自身规则约束：它必须原样写出要搜的字面量。每条规则都由一棵
      合成诱饵树实跑验证，绿灯因此仍然意味着代码干净。
    * 诱饵树放在 ``coffin/.tmp_purity``（已 gitignore），用普通 ``Path.mkdir()``
      创建，用 ``shutil.rmtree(ignore_errors=True)`` 清理。不要换成 ``tempfile``：
      Windows 上以 ``0o700`` 模式创建的目录（``mkdtemp`` 正是如此）会带上沙箱
      令牌删不掉的 DACL，残留物随后会破坏所有人的 ``git status``。
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

#: 本文件的 posix 路径，相对 ``coffin``。按模块 docstring 里所说的理由豁免全部
#: 规则。
SELF = "tests/test_backends_purity.py"

#: 合成诱饵树的临时根目录。放在工作区内并已 gitignore，跑崩了也不会弄脏
#: ``git status``。
BAIT_ROOT = ROOT / ".tmp_purity"

SOURCE_SUFFIXES = frozenset({".py", ".cu", ".c", ".h", ".cpp", ".cc", ".hpp"})

SKIP_DIRECTORIES = frozenset({
    "__pycache__", ".git", ".hg", ".svn", ".venv", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", ".tox", "node_modules",
    ".tmp_purity",
})


@dataclass(frozen=True)
class Pattern:
    """一条识别硬件相关 token 的正则。

    Attributes:
        label: 报告里用来按种类拆分同一规则命中的短名。
        regex: 编译后的表达式；一行中任意位置命中即算违规。
    """

    label: str
    regex: re.Pattern


@dataclass(frozen=True)
class Group:
    """目录树的一个切片，各自带预算，进度才读得出来。

    Attributes:
        name: 随代码发布的用 ``"prod"``，``tests/`` 用 ``"probe"``，后者允许
            最晚保持平台相关。
        prefixes: 该组的目录前缀。空元组让这个组成为兜底组，必须最后声明。
    """

    name: str
    prefixes: tuple[str, ...]


PROD = Group("prod", ())
PROBE = Group("probe", ("tests/",))
SHIPPED = (PROD,)
WITH_PROBES = (PROBE, PROD)


@dataclass(frozen=True)
class Rule:
    """一条纯度规则：扫描范围、分组，以及要往下压的允许列表。

    Attributes:
        name: 稳定的规则 id，用作 unittest 方法名后缀和报告里的名字。
        summary: 一句话说明这条规则保护什么、为什么重要。
        patterns: 识别器；一行命中任意一条即算一个违规行。
        scope: 要扫描的目录前缀。为空表示整棵树。
        exempt: 目录前缀，落在其中的命中照常报告但不计入预算——那正是后端包
            本身。扫描器文件也始终豁免。
        groups: 预算切片，先是 probe，然后是兜底的 ``prod``。
        budgets: 允许列表，形式为 ``组 -> posix 相对路径 -> 精确行数``。
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
    """单条违规行，按编辑器能直接跳转的地址寻址。"""

    path: str
    line: int
    label: str
    text: str

    def render(self) -> str:
        """格式化成 ``path:line: text``，失败清单因此可以直接照着改。

        Returns:
            一行报告，源文本在 200 字符处截断。
        """
        return f"{self.path}:{self.line}: {self.text[:200]}"


@dataclass(frozen=True)
class GroupReport:
    """一条规则中一个组的扫描结果。

    Attributes:
        group: 本切片覆盖的组。
        allowed: 该组生效的允许列表，``路径 -> 行数``。
        per_file: 每个文件的命中数。
        violations: 新增、超预算或低于预算的条目，形式为
            ``(路径, 实际, 允许)``，新文件的 ``允许`` 为 ``None``。
        stale: 已经不再违规的预算键，必须删除。
    """

    group: Group
    allowed: dict[str, int]
    per_file: dict[str, int]
    violations: tuple[tuple[str, int, int | None], ...]
    stale: tuple[str, ...]

    @property
    def budget(self) -> int:
        """该组允许列表仍容忍的违规行总数。"""
        return sum(self.allowed.values())

    @property
    def tracked(self) -> int:
        """该组当前扫到的违规行数。"""
        return sum(self.per_file.values())

    @property
    def ok(self) -> bool:
        """该组没有违规、也没有过期条目时为真。"""
        return not self.violations and not self.stale

    def totals_line(self) -> str:
        """格式化本切片的进度，它就是迁移指标。

        Returns:
            ``"<group>: tracked=<n> budget=<n> files=<n>"``。
        """
        return (f"{self.group.name}: tracked={self.tracked}"
                f" budget={self.budget} files={len(self.per_file)}")


@dataclass(frozen=True)
class RuleReport:
    """一条规则在它所有组上的扫描结果。

    Attributes:
        rule: 产出这份报告的规则。
        scanned: 扫描根下访问到的源文件数。
        hits: 范围内的全部命中，含豁免的部分。
        exempt: 落在规则豁免根下或扫描器里的命中。
        groups: 各组结果，顺序与规则里的声明一致。
    """

    rule: Rule
    scanned: int
    hits: tuple[Occurrence, ...]
    exempt: tuple[Occurrence, ...]
    groups: tuple[GroupReport, ...]

    @property
    def tracked(self) -> int:
        """各组违规行之和。"""
        return sum(group.tracked for group in self.groups)

    @property
    def budget(self) -> int:
        """整份允许列表仍容忍的违规行总数。"""
        return sum(group.budget for group in self.groups)

    @property
    def ok(self) -> bool:
        """所有组都通过时为真。"""
        return all(group.ok for group in self.groups)

    def totals_line(self) -> str:
        """格式化这条规则打印的进度行。

        Returns:
            规则名、各组切片，最后是豁免计数。
        """
        slices = "  ".join(f"[{group.totals_line()}]" for group in self.groups)
        return (f"{self.rule.name}: {slices}  hits={len(self.hits)}"
                f" exempt={len(self.exempt)}")

    def failure_lines(self) -> list[str]:
        """把每条违规和过期条目写成可直接照办的说明行。

        Returns:
            供人阅读的行；规则通过时为空。每个违规文件先给出原因，再跟它的
            ``path:line: text`` 命中行。
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
    """一个文件系统根下所有规则的报告。

    Attributes:
        root: 本次扫描走过的根。
        scanned: 访问到的源文件数。
        reports: 按规则名索引的结果，顺序与规则声明一致。
    """

    root: Path
    scanned: int
    reports: dict[str, RuleReport]

    @property
    def ok(self) -> bool:
        """所有规则都通过时为真。"""
        return all(report.ok for report in self.reports.values())


def _under(path: str, prefixes: tuple[str, ...]) -> bool:
    """判断 posix 相对路径是否落在某个目录前缀之内。

    Args:
        path: 源文件的 posix 相对路径。
        prefixes: 目录前缀，末尾斜杠可有可无。

    Returns:
        ``path`` 等于某个前缀、或以该前缀加 ``/`` 开头时为 ``True``。
    """
    for prefix in prefixes:
        trimmed = prefix.rstrip("/")
        if path == trimmed or path.startswith(trimmed + "/"):
            return True
    return False


def _group_of(rule: Rule, path: str) -> Group:
    """挑出被统计路径所属的组。

    Args:
        rule: 分组取自这条规则。
        path: posix 相对路径。

    Returns:
        前缀包含该路径的第一个组；最后一个组必须是兜底组，格式正确的规则因此
        不会落到无组可归。
    """
    for group in rule.groups:
        if group.prefixes and _under(path, group.prefixes):
            return group
    return rule.groups[-1]


def _iter_sources(root: Path) -> Iterator[tuple[Path, str]]:
    """遍历目录树，逐个产出源文件及其 posix 相对路径。

    遍历本身就是门禁的要点：它必须看见本测试作者从没听说过的文件，因此没有
    后缀或目录清单需要跟着目录树同步。

    Args:
        root: 要遍历的目录。

    Yields:
        每个 ``.py`` / CUDA / C / C++ 源文件对应的
        ``(绝对路径, posix 相对路径)``，按路径排序以保证失败顺序稳定。
    """
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        relative = path.relative_to(root)
        if SKIP_DIRECTORIES.intersection(relative.parts[:-1]):
            continue
        yield path, relative.as_posix()


def _scan_rule(rule: Rule, root: Path) -> tuple[Occurrence, ...]:
    """收集 ``root`` 下某条规则的全部违规行。

    Args:
        rule: 模式与范围取自这条规则。
        root: 扫描根。

    Returns:
        先按路径、再按行号排序的违规行，同一行对同一规则至多记一次。
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
    """对一棵树跑完所有纯度规则。

    Args:
        root: 要扫描的树，默认是 ``coffin``。
        budgets: 可选的预算允许列表覆盖，形式为
            ``规则 -> 组 -> 路径 -> 行数``。``None`` 用声明里的允许列表；传空映射
            等于假装没有允许列表，现有允许列表就是这样生成的。

    Returns:
        一份 :class:`Scan`，每条规则一个报告。
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


# 允许列表。每个条目是该文件当前实际含有的违规行数，按组分开。迁移一步之后用
# ``python tests/test_backends_purity.py --emit-allowlist`` 重新生成；门禁会在
# NEW、OVER、LOOSE 和 STALE 条目上失败，所以这些表只能缩小。
ALLOWLISTS: dict[str, dict[str, dict[str, int]]] = {
    "R1_hip_ops_import": {
        "probe": {
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
            "tests/support.py": 8,
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
            # 导入语句不是唯一的入口：动态导入和直接捅 ``sys.modules`` 都能
            # 摸到 shim 而不留下一行匹配的 import，而这两种形式正是「临时」
            # 绕过会用的写法。
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
            # 相对导入是后端作者手边最顺的写法（``from ..ops import x``），
            # 必须有单独的模式：上面那条绝对路径模式匹配不了开头的点。
            Pattern("relative_layer_import", re.compile(
                r"^\s*from\s+\.{1,3}(?:ops|models|engine|serving|cache)\b")),
        ),
        groups=SHIPPED,
        scope=("backends/",),
        budgets=ALLOWLISTS["R5_backend_dependency"],
    ),
)


def _write_tree(root: Path, files: dict[str, str]) -> None:
    """为诱饵用例实体化一棵合成源码树。

    Args:
        root: 在这些文件要建在其下的目录。
        files: posix 相对路径到文件内容的映射。
    """
    for name, body in files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")


BAIT = {
    "bait_r1.py": "import hip_ops\n",
    # 只看 import 语句的规则会漏掉的绕过方式：动态导入和直接捅
    # ``sys.modules`` 都能摸到 shim 而不留下匹配的 import 行。
    "bait_r1_dyn.py": ('import importlib\n'
                       'SHIM = importlib.import_module("hip_ops")\n'),
    "bait_r1_modules.py": 'import sys\nCACHED = sys.modules["hip_ops"]\n',
    "bait_r2.py": 'ARCH = "gfx906"\nFLAGS = ["--offload-arch=gfx906"]\n',
    "bait_r3.py": ('x = torch.zeros(1, device="cuda")\n'
                   'y = x.cuda()\ntorch.cuda.synchronize()\n'),
    "bait_r4.py": 'HERE = "kernels/gemv.cu"\nRATE = "12 kernels/step"\n',
    # 规则 5 只扫 ``backends/``，诱饵因此必须放在那里。相对导入正是包内作者
    # 下笔时会用的形式。
    "backends/bait_r5.py": "from ops.dtypes import D\nimport models.qwen35\n",
    "backends/bait_r5_rel.py": "from ..ops.dtypes import D\n",
    "__pycache__/junk.py": "torch.cuda.synchronize()\n",
    ".tmp_purity/junk.py": "torch.cuda.synchronize()\n",
}


class PurityGateTest(unittest.TestCase):
    """每条规则都要在当前树上通过，每份允许列表都要是此刻的真实状态。"""

    scan: Scan

    @classmethod
    def setUpClass(cls) -> None:
        """每轮只扫一次，并打印每条规则的合计。

        ``prod`` 切片是必须归零的那个数；合计行则是每条规则各自的迁移进度。
        """
        cls.scan = audit()
        print(f"\n[platform purity] root={cls.scan.root} source_files={cls.scan.scanned}")
        for report in cls.scan.reports.values():
            print(f"[platform purity] {report.totals_line()}")

    def _assert_rule(self, name: str) -> None:
        """规则有违规或过期条目时抛出带可照办说明行的断言失败。

        Args:
            name: :data:`RULES` 里的规则 id。
        """
        report = self.scan.reports[name]
        if report.ok:
            return
        problem = [f"{name}: {report.rule.summary}",
                   f"  ({report.totals_line()})"]
        problem.extend(report.failure_lines())
        self.fail("\n".join(problem))

    def test_rule1_no_hip_ops_import_outside_backend(self):
        """只有 ``backends/`` 可以导入 ``hip_ops``。"""
        self._assert_rule("R1_hip_ops_import")

    def test_rule2_no_platform_build_literals_outside_backend(self):
        """架构字符串和扩展构建开关只属于后端。"""
        self._assert_rule("R2_platform_literal")

    def test_rule3_no_cuda_spelling_outside_backend(self):
        """设备字符串、``.cuda()`` 和 ``torch.cuda`` 都是后端细节。"""
        self._assert_rule("R3_cuda_literal")

    def test_rule4_no_kernels_path_outside_backend(self):
        """后端之外的调用方一律不许指名 ``kernels/`` 下的文件。"""
        self._assert_rule("R4_kernels_path")

    def test_rule5_backends_do_not_import_upward(self):
        """后端包不许反向导入依赖它的那些层。"""
        self._assert_rule("R5_backend_dependency")

    def test_scan_reaches_the_known_tree(self):
        """遍历必须看见真实的树，不能是空的或残缺的。

        防的是扫描根悄悄指错目录：那样门禁什么都没看也会通过。断言的对象是
        *遍历*访问到的文件，不是有命中的文件。以命中为基准会把目录布局和迁移
        自身的进度一并钉死：内核搬进后端之后 ``hip_ops.py`` 就不再存在，
        ``models/qwen35/model.py`` 的设备字面量被替换后也就不再违规——树更干净
        不该让这个测试失败。
        """
        self.assertGreater(self.scan.scanned, 100)
        walked = {path.relative_to(self.scan.root).as_posix()
                  for path, _ in _iter_sources(self.scan.root)}
        for expected in ("models/qwen35/model.py", "tests/support.py",
                         "backends/reference/__init__.py", "ops/gemv.py"):
            self.assertIn(expected, walked)

    def test_production_is_separated_from_probe_directories(self):
        """规则 3 必须把随代码发布的代码与 ``tests/`` 分开报告。

        否则一条只对测试放宽的许可会把回归的生产文件遮过去，而迁移要压到零的
        正是生产那个数。

        生产现在已经是零，所以断言换成：拆分还在，且 ``prod`` 组里没有任何路径
        落在探针目录下。以前还要求 ``prod`` 非空，那等于钉住迁移自己的进度，
        恰恰会在它努力奔向的状态上失败。
        """
        report = self.scan.reports["R3_cuda_literal"]
        groups = {group.group.name: group for group in report.groups}
        self.assertEqual(set(groups), {"prod", "probe"})
        self.assertTrue(all(not path.startswith("tests/")
                            for path in groups["prod"].per_file))
        self.assertTrue(all(path.startswith("tests/")
                            for path in groups["probe"].per_file))
        # v3 规则同名的性质：已发布的文件一律不许再写出厂商的设备 API。探针
        # 可以，而且确实还在写。
        self.assertEqual(groups["prod"].tracked, 0,
                         "shipped code must not name one vendor's device stack")

    def test_rules_catch_synthetic_offenders(self):
        """每条规则都要在诱饵树上触发，绿灯才等于代码干净。

        诱饵同时钉住噪音处理：``__pycache__`` 下或门禁自己临时根下的违规文件
        必须忽略，``12 kernels/step``（一个速率，不是路径）不许算作内核引用。
        预算那几组断言钉住门禁能给出的四种判决。
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
        """合并一条规则在各组的命中数。

        Args:
            scan: 扫描结果。
            name: 规则 id。

        Returns:
            跨所有组的 ``路径 -> 违规行数``。
        """
        merged: dict[str, int] = {}
        for group in scan.reports[name].groups:
            merged.update(group.per_file)
        return merged

    @staticmethod
    def _violations(scan: Scan, name: str) -> tuple[tuple[str, int, int | None], ...]:
        """收集一条规则在各组的全部违规。

        Args:
            scan: 扫描结果。
            name: 规则 id。

        Returns:
            ``(路径, 实际, 允许)`` 元组。
        """
        return tuple(violation for group in scan.reports[name].groups
                     for violation in group.violations)


def format_allowlist(scan: Scan) -> str:
    """把当前每文件命中数渲染成允许列表字面量。

    Args:
        scan: 用空预算审出来的扫描结果。

    Returns:
        :data:`ALLOWLISTS` 的 Python 源码，已排序，可直接粘贴。
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
    """打印审计结果，或者打印与当前树相符的允许列表。

    树是红的时 ``--emit-allowlist`` 拒绝运行。从红树上重新生成预算，等于把当前
    违规固化成新基线，那是允许列表唯一还能长大的途径；先要求绿灯，就把这件事变成
    作者必须处理的错误，而不是默默接受。

    Args:
        argv: 命令行参数；``--report`` 或 ``--emit-allowlist``。

    Returns:
        进程退出码：树违反规则时（或者在红树上要 ``--emit-allowlist`` 时）为 1，
        其余为 0。
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