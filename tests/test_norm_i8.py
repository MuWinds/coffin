"""融合 RMSNorm 直出 int8 的两条路径必须与「拆开跑」逐位相同。

拆开的写法是 ``rms_norm`` / ``add_rms_norm`` 把归一化结果写成 bf16，再由
``quant_act_i8`` 读回来量化。融合版把这些收进一次启动、int8 直出，省掉两趟
global 往返（预填充一步实测省 5.96 ms）。

这里守的不变量是**逐位相等**，不是「误差很小」。理由有两条，都不是洁癖：

* 融合版只是把同一串运算挪进一个内核，算术上没有任何近似。一旦不等，说明多做了
  或漏做了一次舍入，而那条错误会穿过 32 层被放大——它不会以「慢」或「崩」的形式
  出现，只会让某些 token 悄悄变掉。
* 这条路径真正的坑正是**一次多出来的舍入**：归一化结果必须先舍入成 bf16 再取
  absmax，因为拆开跑时 ``quant_act_i8`` 读到的就是已经舍入的那一份。用未舍入的
  fp32 取 absmax 会让 scale 差几个 ULP、贴近边界的 code 差 1，签名是
  ``max|dq|=1.0``、``max|ds|~1e-4`` 而**残差本身完全相同**。容差断言看不见这种差异，
  只有 ``torch.equal`` 抓得住。

Module: ``coffin/tests``。只经 ``backends.ops`` 调内核，不加载 checkpoint。

Constraints:
    * 没有原生内核的后端（CPU / reference）上整类跳过：那条路径没有融合算子，
      测它等于测参考实现自己。
    * 每条断言都比三样东西——int8 codes、每行 scale、以及被原地累加的残差——
      因为三者任一不同都会让下游算出别的数，而只看 codes 会漏掉 scale 的 ULP 偏差。
"""

from __future__ import annotations

import unittest

import support

import torch

import backends

#: 覆盖解码（M=1）、小批（M=2/64）和预填充分块（M=2048）四种块数。线程数的最优点
#: 由 M 决定，所以这四档才把两条分派分支都走到。
_ROWS = (1, 2, 64, 2048)

#: 4B 的 hidden，也是这两条内核上最大的真实行宽（LDS 预算是按它验的）。
_HIDDEN = 2560

#: 设备字符串从后端取，不写字面量：纯度门禁的 R3 规则就是为了拦住「测试里写死某一家
#: 的栈」，而那正是同一份源码要在两张卡上跑时要避免的东西。
_DEV = backends.device


def _has_native_norm() -> bool:
    """两个融合入口是否都由原生内核承担。

    注意 ``backends.ops`` 是**已解析的表实例**，不是工厂函数 —— 写成
    ``backends.ops()`` 会 ``TypeError``，而那个异常被上一行的
    ``device == "cpu"`` 提前返回盖住：坏守卫在本地永远是「跳过」，在卡上永远让整个
    模块 import 失败。两个结果都错，它既没有保护什么也没有报出问题。

    判据只能直接问扩展：这两个融合入口不是 :class:`~backends.protocol.Ops` 的条目
    （它们是可选优化，不该逼每个后端都提供一个假实现），所以表里查不到。
    """
    if backends.caps().device == "cpu":
        return False
    try:
        from backends.corex import kernels

        module = kernels.load_kernels()
    except Exception:                            # noqa: BLE001 - no toolchain is a skip
        return False
    return hasattr(module, "rms_norm_i8") and hasattr(module, "add_rms_norm_i8")


@unittest.skipUnless(_has_native_norm(), "needs a backend with native fused norm kernels")
class FusedNormInt8Test(unittest.TestCase):
    """归一化直出 int8 的两条路径，对拆开跑的两内核版本逐位相等。"""

    def setUp(self) -> None:
        """取本后端的扩展模块，并固定一组量级真实的输入。"""
        from backends.corex import kernels

        self.module = kernels.load_kernels()
        # 全零输入会让 absmax 落到 1e-5 下限，把量化的舍入差异掩盖掉，所以输入必须
        # 有真实量级；增益取零中心区间，与 checkpoint 里 ``1 + w`` 折完的范围相当。
        g = torch.Generator(device=_DEV).manual_seed(20261031)
        self.base = torch.randn(_ROWS[-1], _HIDDEN, dtype=torch.bfloat16,
                                device=_DEV, generator=g)
        self.gain = (torch.rand(_HIDDEN, dtype=torch.bfloat16, device=_DEV,
                                generator=g) * 1.8 - 0.8)

    def _rows(self, rows: int):
        """切出前 ``rows`` 行，让四档共用同一批数字而不是四批独立采样。"""
        return self.base[:rows]

    def test_rms_norm_i8_matches_the_two_kernel_chain(self):
        """预填充那条：``rms_norm`` 写 bf16 再量化，与直出 int8 逐位相同。"""
        for rows in _ROWS:
            with self.subTest(rows=rows):
                x = self._rows(rows)
                n = self.module.rms_norm(x, self.gain, 0, 1e-6)
                q_ref = torch.empty(rows, _HIDDEN, dtype=torch.int8, device=_DEV)
                s_ref = torch.empty(rows, dtype=torch.float32, device=_DEV)
                self.module.quant_act_i8(n, q_ref, s_ref)

                q_new = torch.empty(rows, _HIDDEN, dtype=torch.int8, device=_DEV)
                s_new = torch.empty(rows, dtype=torch.float32, device=_DEV)
                self.module.rms_norm_i8(x, self.gain, q_new, s_new, 0, 1e-6)

                self.assertTrue(torch.equal(q_ref, q_new),
                                f"int8 codes differ at M={rows}")
                self.assertTrue(torch.equal(s_ref, s_new),
                                f"row scale differs at M={rows}")

    def test_add_rms_norm_i8_matches_the_two_kernel_chain(self):
        """解码那条：``add_rms_norm`` 写 bf16 再量化，与直出 int8 逐位相同。

        残差被原地累加，所以两条臂必须各拿一份自己的残差副本——共用一份会让第二条
        臂读到已经被加过一次的值，而那种差异比量化误差大得多，会把这条断言变成
        一个永远失败的噪声源。
        """
        for rows in _ROWS:
            with self.subTest(rows=rows):
                x = self._rows(rows)
                mix = torch.randn_like(x) * 0.1

                x_ref = x.clone()
                n_ref = torch.empty(rows, _HIDDEN, dtype=torch.bfloat16, device=_DEV)
                self.module.add_rms_norm(x_ref, mix, self.gain, n_ref, 0, 1e-6)
                q_ref = torch.empty(rows, _HIDDEN, dtype=torch.int8, device=_DEV)
                s_ref = torch.empty(rows, dtype=torch.float32, device=_DEV)
                self.module.quant_act_i8(n_ref, q_ref, s_ref)

                x_new = x.clone()
                q_new = torch.empty(rows, _HIDDEN, dtype=torch.int8, device=_DEV)
                s_new = torch.empty(rows, dtype=torch.float32, device=_DEV)
                self.module.add_rms_norm_i8(x_new, mix, self.gain, q_new, s_new, 0, 1e-6)

                self.assertTrue(torch.equal(q_ref, q_new),
                                f"int8 codes differ at M={rows}")
                self.assertTrue(torch.equal(s_ref, s_new),
                                f"row scale differs at M={rows}")
                # 残差必须仍然被原地累加：下游层靠它继续往下加。
                self.assertTrue(torch.equal(x_ref, x_new),
                                f"residual differs at M={rows}")

    def test_quant_act_i8_round_trips_within_one_quantisation_step(self):
        """激活量化器自身的上界：反量化误差不得超过一个量化步长的半个。

        这条钉住的是 scale 约定（``absmax / 127``），不是精度水平。约定错了
        （例如按张量而不是按行、或除 128 而不是 127）会让误差整体偏一档，而逐位
        测试只比较两条实现是否一致，对「两条一起错」不敏感。
        """
        rows = 64
        x = self._rows(rows)
        q = torch.empty(rows, _HIDDEN, dtype=torch.int8, device=_DEV)
        s = torch.empty(rows, dtype=torch.float32, device=_DEV)
        self.module.quant_act_i8(x, q, s)
        deq = q.float() * s[:, None]
        step = s.repeat_interleave(1).unsqueeze(1)
        err = (deq - x.float()).abs()
        self.assertTrue(bool((err <= step * 0.5 + 1e-6).all()),
                        f"max err {float(err.max()):.6f} exceeds half a quantisation step")


if __name__ == "__main__":
    unittest.main()
