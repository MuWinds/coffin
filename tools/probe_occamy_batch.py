"""对拍 Occamy 单流生成与 4/8 行并发生成的 token 序列。

只用于 SCOW 临时诊断，不进入测试目录：它加载真实 Occamy 权重，并通过运行时
socket 同时提交长短请求，让已结束行在其他行仍解码时被重新分配。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys


def _generate(sock: str, prompt: str, max_new: int) -> list[int]:
    """请求贪心生成并返回 token id。"""
    command = [sys.executable, "-B", "-u", "-m",
               "entrypoints.cli.runtime", "--sock", sock, "generate",
               "--prompt", prompt, "--max-new", str(max_new),
               "--temperature", "0", "--print-ids"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=900)
    if result.returncode != 0:
        raise RuntimeError(
            f"exit={result.returncode} stdout={result.stdout[-1200:]!r} "
            f"stderr={result.stderr[-1200:]!r}")
    line = next((line for line in reversed(result.stdout.splitlines())
                 if line.startswith("IDS ")), None)
    if line is None:
        raise RuntimeError(f"generation returned no token ids: {result.stdout[-500:]}")
    return json.loads(line[4:])


def _first_diff(left: list[int], right: list[int]) -> int | None:
    """返回首个不同 token 的位置；两份序列相同时返回 ``None``。"""
    for i, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return i
    return min(len(left), len(right)) if len(left) != len(right) else None


def main() -> int:
    """顺序生成基线，再并发提交同一组长短请求。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sock", required=True)
    parser.add_argument("--level", type=int, choices=(4, 8), required=True)
    args = parser.parse_args()

    body = (
        "A field engineer records the measured temperature, pressure, and flow rate. "
        "Each reading is checked against the previous sample and the calibration log. "
    ) * 24
    cases = [(f"Case {i}: station {i} reports a routine system inspection. {body}",
              (8, 24, 48, 64)[i] if i < 4 else 96)
             for i in range(12)]

    baseline = []
    for i, (prompt, limit) in enumerate(cases):
        try:
            baseline.append(_generate(args.sock, prompt, limit))
        except Exception as exc:
            print(f"BASELINE_ERROR case={i} {type(exc).__name__}: {exc}",
                  flush=True)
            return 2
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(cases)) as pool:
        futures = [pool.submit(_generate, args.sock, prompt, limit)
                   for prompt, limit in cases]
        batched = [None] * len(futures)
        client_errors = 0
        for i, future in enumerate(futures):
            try:
                batched[i] = future.result()
            except Exception as exc:
                client_errors += 1
                print(f"CLIENT_ERROR case={i} {type(exc).__name__}: {exc}",
                      flush=True)

    passed = 0
    for i, (expected, actual) in enumerate(zip(baseline, batched)):
        if actual is None:
            continue
        mismatch = _first_diff(expected, actual)
        if mismatch is None:
            passed += 1
            print(f"case={i} PASS tokens={len(expected)}", flush=True)
        else:
            left = expected[mismatch] if mismatch < len(expected) else "END"
            right = actual[mismatch] if mismatch < len(actual) else "END"
            print(f"case={i} FAIL first_diff={mismatch} baseline={left} batch={right} "
                  f"lengths={len(expected)}/{len(actual)}", flush=True)
    print(f"RESULT level={args.level} pass={passed}/{len(cases)} "
          f"client_errors={client_errors}", flush=True)
    return 0 if passed == len(cases) and client_errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
