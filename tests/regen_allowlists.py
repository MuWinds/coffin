"""按代码树当前的实际情况重新生成纯度允许列表。

门禁的规则是预算表只许收缩，而且每次收缩必须和造成收缩的改动落在同一次提交
里，否则数字会与现实脱节，门禁就不再说明任何问题。一次清理违规的迁移之后，
代码树会为几十个文件报出 ``LOOSE``（预算高于实际）和 ``STALE``（条目对应的
文件已不再违规），手工下调这些数字正是抄写错误的来源。

所以这个工具复用门禁自己的扫描器：代码树里存在 ``NEW`` 违规（完全没有预算的
违规，是真正的回归，重新生成会把它悄悄放行）时它拒绝运行；否则把字面量改写
成扫描器看到的样子。

``LOOSE`` 和 ``STALE`` 不是回归，它们是门禁在要求下调预算，而这正好是本工具
做的事。

Usage::

    python tests/regen_allowlists.py            # 试运行，打印新表
    python tests/regen_allowlists.py --write    # 原地重写字面量

Module: ``coffin/tests``。维护工具而非测试：``unittest discover`` 只匹配
``test_*.py``，所以本文件不会被收集。

Constraints:
    * 只负责抄写，不负责判断：某个违规该不该被允许由门禁的规则决定，这里只把
      扫描结果落进预算表。
    * 一旦出现 ``NEW`` 违规就整轮拒绝写入，避免把回归洗成新的基线。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# 门禁与它同目录，按模块名导入前必须先把 tests/ 放进 sys.path；所以这条
# import 不能挪到文件顶部，下面那条工具指令压的正是放置顺序告警。
import test_backends_purity as gate  # noqa: E402

#: 门禁用来标记「没有任何预算条目的违规」的字符串。其余标记（``OVER``、
#: ``LOOSE``、``STALE``）都是预算与代码树不一致，而那正是重新生成要处理的。
REGRESSION_MARKER = "NEW"

#: 本工具要改写的字面量。按完整签名定位，避免命中别处同名的字符串。
_ANCHOR = "ALLOWLISTS: dict[str, dict[str, dict[str, int]]] = {"


def regressions(scan: gate.Scan) -> list[str]:
    """挑出没有预算覆盖的违规，作为拒绝重新生成的依据。

    Args:
        scan: 按当前允许列表审计过的扫描结果。

    Returns:
        每条 ``NEW`` 违规一行。非空表示树里有被新引入而没有预算的违规，
        此时不能覆盖生成预算表，否则等于给它发通行证。
    """
    out = []
    for name, report in scan.reports.items():
        for line in report.failure_lines():
            if line.lstrip().startswith(REGRESSION_MARKER):
                out.append(f"{name}: {line.strip()}")
    return out


def main(argv: list[str] | None = None) -> int:
    """打印或写入与当前代码树一致的允许列表。

    Args:
        argv: 命令行参数；``--write`` 表示原地改写文件。

    Returns:
        进程退出码：树里有重新生成会放行的回归时为 1，否则为 0。
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

    # 每条规则都给一份空预算，这样每个命中都会暴露出来——``format_allowlist``
    # 正是据此从零重建整张表。
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
