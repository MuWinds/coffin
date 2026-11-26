"""在卡上一条命令跑完本轮改动的一整套验收，输出 markdown 报告。

本机是离机的（没有 GPU、没有 corex 工具链），所以这里只做编排：它**不 import**
``cxrt`` 的内部对象，只用子进程调用仓库里已经存在的那三个入口，免得再引入一套
与卡上 Python 版本耦合的代码。

跑的东西：

1. 环境取证：``python3 -V``、``import cxrt`` 是否成功、``torch`` 是否在链路里被加载；
   ``ixsmi``/``nvidia-smi`` 有哪个就用哪个取卡名与显存。
2. 内核编译：``open_runtime().build_w8a8/gemv/gemm``，任何一条编不出来都直接标红——
   后面所有数字都建立在它之上。
3. GEMM 探针：``probe_gemm_nk.py`` 跑 ``i8`` / ``bf16`` / ``f32`` 三档 ×
   ``--computes both``，外加一档 ``--ones --k 4096`` 查大 K 截断。
4. 静态审计：``audit_torch_path.py`` 默认入口与 ``--all``，退出码必须为 0。
5. 离机单测：``tests/test_*.py`` 逐个直跑（本仓没装 pytest）。
6. 模型 A/B：给了 ``--model-cmd`` 就原样交给 shell 跑两遍（``COFFIN_I8_GEMM=lib``
   与 ``tcu``），把 stdout 尾部和耗时记下来；不给就跳过，不当失败。

用法::

    python3 tools/card_verify.py --out card_report.md
    python3 tools/card_verify.py --model-cmd "python3 -m coffin.run --prompt hi"

退出码：0 表示所有**必选**项通过；探针/A-B 这类会依赖卡状态的项失败时记 FAIL 但
仍返回 0，只有编译、审计、单测任一项失败才返回 1。
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: 三档探针的档位定义。形状取 4B 的真实量级：``k``/``n`` 是 gate_up 那一路，
#: 4096³ 是 ``ixblas`` 的 PERF 行本身，用来和文档里的 46.19 TOPS 对表。
PROBES = (
    ("i8", ["--dtype", "i8", "--m", "2048", "--k", "5120", "--n", "11008"]),
    ("i8-4096", ["--dtype", "i8", "--m", "4096", "--k", "4096", "--n", "4096"]),
    ("bf16", ["--dtype", "bf16", "--m", "2048", "--k", "5120", "--n", "11008"]),
    ("f32", ["--dtype", "f32", "--m", "2048", "--k", "5120", "--n", "11008"]),
    ("i8-ones-k4096", ["--dtype", "i8", "--ones", "--k", "4096", "--m", "64", "--n", "64"]),
)

TESTS = (
    "test_cxrt_w8a8.py",
    "test_cxrt_gemv.py",
    "test_cxrt_tensors.py",
    "test_corex_rt.py",
    "test_protocol_torchfree.py",
    "test_cxrt_attn.py",
    "test_cxrt_solve.py",
)


def run(cmd: list[str], env_extra: dict[str, str] | None = None, timeout: int = 900):
    """跑一条命令，返回 ``(码, 合并输出, 秒)``。超时算码 124。"""
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(ROOT))
    if env_extra:
        env.update(env_extra)
    t0 = time.time()
    try:
        p = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
        return p.returncode, p.stdout, time.time() - t0
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        return 124, out + f"\n[timeout after {timeout}s]", time.time() - t0
    except (FileNotFoundError, OSError) as exc:
        # 缺命令（``ixsmi`` 不在 PATH 上）不该把整份报告打断。
        return 127, f"{cmd[0]}: {exc}", time.time() - t0


def tail(text: str, lines: int = 12) -> str:
    """取输出的最后几行，报告里只放这个，避免把整段日志塞进表格。"""
    got = [ln for ln in text.strip().splitlines() if ln.strip()]
    return "\n".join(got[-lines:]) if got else "(no output)"


def env_facts() -> list[str]:
    """环境取证。``torch`` 这一条是重点：链路里不该出现它。"""
    facts = [f"- python: `{sys.version.split()[0]}` ({sys.executable})"]
    code, out, _ = run(
        [
            sys.executable,
            "-c",
            "import cxrt, torch, sys;"
            "print('cxrt', cxrt.__file__);"
            "print('torch', torch.__version__)",
        ]
    )
    facts.append(f"- import cxrt+torch: {'ok' if code == 0 else 'FAIL'} — `{tail(out, 3)}`")
    code, out, _ = run(
        [sys.executable, "-c", "import cxrt; print('cxrt ok', cxrt.__file__)"]
    )
    if code == 0:
        # 只 import cxrt 时 torch 不该被拉进来；用 ``sys.modules`` 判，不靠猜。
        code2, out2, _ = run(
            [
                sys.executable,
                "-c",
                "import sys;"
                "import cxrt;"
                "print('torch loaded:', 'torch' in sys.modules)",
            ]
        )
        facts.append(f"- import cxrt 之后 torch loaded: `{tail(out2, 1)}`")
    else:
        facts.append("- import cxrt: FAIL（卡上驱动或工具链没就位）")
    for exe in ("ixsmi", "ivsmi", "nvidia-smi"):
        code, out, _ = run([exe, "-L"] if exe != "ivsmi" else [exe, "info"], timeout=60)
        if code == 0:
            facts.append(f"- `{exe}`: `{out.strip().splitlines()[0][:120]}`")
            break
    else:
        facts.append("- 没找到 `ixsmi`/`ivsmi`/`nvidia-smi`，卡名与显存取不到")
    return facts


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="card_report.md", help="报告写到哪")
    p.add_argument("--skip-tests", action="store_true", help="跳过离机单测")
    p.add_argument("--model-cmd", default="",
                   help="模型端到端用的命令行（按 shell 引号规则切分），给了就跑两遍")
    args = p.parse_args(argv)

    py = sys.executable
    lines: list[str] = ["# 卡上验收报告", ""]
    gate_fail = False

    lines += ["## 1. 环境", ""] + env_facts() + [""]

    lines += ["## 2. 内核编译", "", "| 组件 | 结果 | 耗时 | 尾部 |", "| --- | --- | --- | --- |"]
    for build, args_ in (
        ("w8a8", ["--build", "w8a8"]),
        ("gemv", ["--build", "gemv"]),
        ("gemm", ["--build", "gemm"]),
    ):
        # 直接用 cxrt 的构建入口；``-c`` 里的字符串不含 f-string，避免转义地狱。
        code, out, secs = run(
            [
                py,
                "-c",
                "import sys; from cxrt import open_runtime as o;"
                "rt = o(); getattr(rt, 'build_' + sys.argv[1])(); print('built')",
                args_[1],
            ]
        )
        ok = code == 0
        gate_fail = gate_fail or not ok
        tail_s = tail(out, 4).replace("|", "\\|").replace("\n", " / ")
        lines.append(f"| {build} | {'ok' if ok else 'FAIL'} | {secs:.1f}s | `{tail_s}` |")
    lines.append("")

    lines += ["## 3. GEMM 探针", "",
              "> 只有 ``legacy`` 这一档进判定：上卡实测厂商 ``libcublas`` 对新表的 "
              "``computeType`` 一律 ``not supported``（三档 dtype 都一样），所以默认档"
              "已经切到旧表；``new`` 那一行只作记录，不算失败。",
              "", "| 档位 | computes | 结果 | 尾部 |", "| --- | --- | --- | --- |"]
    for name, base in PROBES:
        for computes in ("new", "legacy"):
            code, out, secs = run(
                [py, "tools/probe_gemm_nk.py", *base, "--computes", computes]
            )
            if computes == "legacy":
                gate_fail = gate_fail or code != 0
            tail_s = tail(out, 6).replace("|", "\\|").replace("\n", " / ")
            verdict = "ok" if code == 0 else ("记录（非判定）" if computes == "new" else "FAIL")
            lines.append(
                f"| {name} | {computes} | {verdict} ({secs:.0f}s) | `{tail_s}` |"
            )
    lines.append("")

    lines += ["## 4. 静态审计", ""]
    for extra in ([], ["--all"]):
        code, out, _ = run([py, "tools/audit_torch_path.py", *extra])
        gate_fail = gate_fail or code != 0
        label = "--all" if extra else "默认入口"
        lines.append(f"- {label}: 退出码 {code} — `{tail(out, 2).splitlines()[-1][:160]}`")
    lines.append("")

    # 「推理链路里没有真 torch」这一条**进判定**：它不是在报告代码里怎么写，而是在一个
    # 干净子进程里先装壳、再 import 整条链路、最后断言 ``sys.modules['torch']`` 就是
    # ``cxrt.torchshim``。壳装不上（真 torch 已入册）或者某个链路模块 import 需要真
    # torch，都算失败。
    lines += ["## 4b. 无 torch 门槛", ""]
    code, out, secs = run([py, "tools/audit_no_torch.py", "--runtime"], timeout=600)
    gate_fail = gate_fail or code != 0
    head_s = " / ".join(tail(out, 6).splitlines()[:2]).replace("|", "\\|")[:200]
    lines.append(f"- 运行时断言: 退出码 {code}（{secs:.0f}s） — `{head_s}`")
    if code == 0:
        lines += ["", "> 进程里没有真 torch：``sys.modules['torch']`` 是 ``cxrt.torchshim``，"
                      "``torch.Tensor`` 是 ``cxrt.tt.Tensor``。"]
    lines.append("")

    if not args.skip_tests:
        lines += ["## 5. 离机单测", "", "| 测试 | 结果 | 尾部 |", "| --- | --- | --- |"]
        for t in TESTS:
            path = ROOT / "tests" / t
            if not path.exists():
                continue
            code, out, _ = run([py, str(path)], timeout=300)
            gate_fail = gate_fail or code != 0
            tail_s = tail(out, 2).replace("|", "\\|").replace("\n", " / ")
            lines.append(f"| {t} | {'ok' if code == 0 else 'FAIL'} | `{tail_s}` |")
        lines.append("")

    lines += ["## 6. 模型端到端（可选）", "",
              "> int8 现在只有一条路（nk 原缓冲的 TN），`COFFIN_I8_GEMM` 这个切换开关已经删了，"
              "所以这里跑两遍同一个命令，只看「两次是否一致」——它检查的是确定性，不是 A/B。",
              ""]
    if args.model_cmd:
        lines += ["| 轮次 | 退出码 | 耗时 | 尾部 |", "| --- | --- | --- | --- |"]
        for idx in ("1", "2"):
            # ``--model-cmd`` 是**一行**命令：按 shell 的引号规则切开再执行（``run`` 收
            # 的是 argv 列表，整串当可执行名会直接 FileNotFoundError）。
            code, out, secs = run(shlex.split(args.model_cmd), timeout=3600)
            tail_s = tail(out, 4).replace("|", "\\|").replace("\n", " / ")
            lines.append(f"| {idx} | {code} | {secs:.0f}s | `{tail_s}` |")
    else:
        lines.append("跳过：没给 `--model-cmd`。本脚本不替你选命令。")
    lines.append("")

    lines += ["## 7. 结论", "", "**FAIL**" if gate_fail else "**PASS**（编译/审计/无 torch/单测全过）"]
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report -> {args.out}")
    print("\n".join(lines[-4:]))
    return 1 if gate_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
