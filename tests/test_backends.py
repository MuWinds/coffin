"""平台后端层的一致性测试与选择测试。

这里守两件事，且都要在没有任何加速卡的主机上成立：

* 选择是确定的、可读的：``COFFIN_BACKEND`` 优先，未知或不可用的名字直接报错而
  不是悄悄降级，纯 torch 的参考实现永远是最后一个候选；
* 参考后端实现了全部 23 个算子，因为 ``hip_ops.available()`` 分支删掉之后，引擎
  其余部分就靠它兜底。这里留一个 ``None``，就是一个只会在不支持的平台上以请求
  中途崩溃的形式出现的洞。

Module: ``coffin/tests``。导入 ``support``（用于 ``sys.path`` 设置）、``backends``
与 ``int8_weights``；不加载 checkpoint、不要求 DCU，所以纯 CPU 的 CI 步骤和容器
里都能跑。

Constraints:
    * 对照实现与被测后端代码分开写（delta 递推、int8 KV 缓存的 bf16 扩宽、深度
      卷积）。拿后端自己的辅助函数当参照，只能证明这个辅助函数和自己一致。
    * 容差按 dtype 定，不是逐算子拍脑袋：fp32 比较精确到几个 ulp，bf16 比较放行
      bf16 的量化步长（约最大幅值的 0.8%）。对 bf16 结果卡更紧的界等于重新质疑
      这个 dtype。
    * 走环境变量的路径在子进程里跑。进程内改 ``os.environ`` 会漏进已经选好的
      后端表，那张表按设计在进程生命周期内被缓存；子进程也是断言「导入这个包
      不会编译内核」的唯一办法。
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import unittest
from pathlib import Path

# 这条 import 有副作用：它设好 MKL_THREADING_LAYER 并把 coffin/ 放进 sys.path，
# 后面所有 import 都依赖它已经生效，所以不能删也不能挪。
import support  # noqa: F401

import torch
import torch.nn.functional as F

import backends
import backends.hip
import backends.reference
from backends.protocol import Caps, Ops, make_device
from int8_weights import QWeight

#: ``coffin/`` 自身。子进程要以同一个导入根启动。
ENGINE_DIR = Path(__file__).resolve().parent.parent

#: bf16 结果的相对容差：这是 dtype 自己的量化步长，不是错误预算。放宽到超过它，
#: 就会掩盖真正错的归约顺序。
BF16_TOL = 2e-2

#: fp32 结果的相对容差，即重排求和后的几个 ulp。
FP32_TOL = 1e-5


def _rel(got: torch.Tensor, ref: torch.Tensor) -> float:
    """以参照幅值归一化后的最大绝对差。

    相对指标在任何层深下读到的数都一样；绝对界得按形状重新调，那正是容差变得
    毫无意义的由来。
    """
    return ((got.float() - ref.float()).abs().max() / ref.float().abs().max().clamp_min(1e-12)).item()


def _run_py(code: str, env: dict | None = None) -> subprocess.CompletedProcess:
    """在 ``coffin/`` 为根目录的新解释器里跑一段代码。

    Args:
        code: 要执行的 Python 源码。
        env: 覆盖在当前环境之上的环境变量条目。

    Returns:
        执行完的进程，stdout/stderr 已按文本捕获。
    """
    full = dict(os.environ)
    full.pop("COFFIN_BACKEND", None)
    if env:
        full.update(env)
    return subprocess.run([sys.executable, "-c", code], cwd=str(ENGINE_DIR),
                          capture_output=True, text=True, env=full, timeout=180)


class OpTableTest(unittest.TestCase):
    """``Ops`` 就是契约：23 个名字、固定顺序、不许有被悄悄丢掉的条目。"""

    def test_names_are_the_contract_ops(self):
        """改名或漏名对每个调用点都是契约破坏，所以元组按字面钉死。"""
        self.assertEqual(Ops.NAMES, (
            "linear", "linear_m", "linear_prefill",
            "rms_norm", "add_rms_norm", "rms_norm_gated", "kv_quant", "silu_mul",
            "rope_apply",
            "attn_decode", "attn_paged", "attn_paged_batch", "attn_paged_multi_q",
            "conv1d_silu", "conv1d_silu_seq", "conv1d_silu_batch",
            "delta_prologue", "delta_prologue_batch", "delta_prep_seq",
            "delta", "delta_batch", "delta_normed",
            "triangular_solve",
        ))
        self.assertEqual(len(Ops.NAMES), 23)

    def test_names_match_the_dataclass_fields_in_order(self):
        """这条元组是手工维护的，可能与 dataclass 漂移；这里就是漂移报警器。"""
        from dataclasses import fields

        self.assertEqual(tuple(f.name for f in fields(Ops)), Ops.NAMES)

    def test_default_table_is_all_none(self):
        """``None`` 是原生后端表示「没有内核」的方式，空表必须恰好是那个样子。"""
        table = Ops()
        self.assertEqual(table.as_dict(), {name: None for name in Ops.NAMES})
        self.assertEqual(table.implemented(), ())
        self.assertEqual(table.missing(), Ops.NAMES)

    def test_from_mapping_rejects_unknown_names(self):
        """拼错一个名字不能换来一张悄悄缺了该算子的表；那和「平台没有这个内核」看起来一模一样。"""
        with self.assertRaises(ValueError):
            Ops.from_mapping({"gemv": lambda: None})
        with self.assertRaises(ValueError):
            Ops.from_mapping({"linear": lambda: None, "Linear": lambda: None})

    def test_post_init_rejects_a_non_callable_entry(self):
        """登记张量或字符串会在第一次 decode 步才炸，而且只在某一个形状上炸。"""
        with self.assertRaises(TypeError):
            Ops(linear=1)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            Ops(rms_norm="not a function")  # type: ignore[arg-type]

    def test_from_mapping_is_partial_and_round_trips(self):
        """原生后端只列自己写了的内核；其余必须回成 ``None``，留给解析器补齐。"""
        fn = lambda *_a, **_k: None
        table = Ops.from_mapping({"linear": fn})
        self.assertIs(table.linear, fn)
        self.assertEqual(table.implemented(), ("linear",))
        self.assertEqual(set(table.missing()), set(Ops.NAMES) - {"linear"})


class CapsTest(unittest.TestCase):
    """能力记录只装模型代码真正据以分支的事实，而且必须能解析成设备。"""

    def test_make_device_resolves_cpu_without_an_index(self):
        """``torch.device("cpu", 0)`` 在各 torch 版本间不可移植，所以 CPU 路径必须丢掉索引。"""
        self.assertEqual(make_device(Caps(name="x", device="cpu", device_index=3)),
                         torch.device("cpu"))

    def test_make_device_resolves_an_indexed_accelerator(self):
        """加速卡上的每次分配都必须落在 ``<type>:<index>``；只给类型字符串会跟着当前设备走。"""
        self.assertEqual(make_device(Caps(name="x", device="cuda", device_index=1)),
                         torch.device("cuda", 1))
        self.assertEqual(make_device(Caps(name="x", device="xpu", device_index=0)),
                         torch.device("xpu", 0))

    def test_make_device_rejects_an_unknown_type(self):
        """后端 caps 里的拼写错误必须在选择时失败，而不是拖到模型构建器里第一次分配。"""
        with self.assertRaises((RuntimeError, ValueError)):
            make_device(Caps(name="x", device="not-a-device"))

    def test_native_is_a_subset_of_the_contract(self):
        """banner 打印 ``len(native)/len(Ops.NAMES)``，所以契约外的名字会让这个比值说谎。"""
        for candidate in backends.candidates():
            caps = candidate.backend.caps()
            self.assertTrue(caps.native <= set(Ops.NAMES),
                            f"{caps.name}: {sorted(caps.native - set(Ops.NAMES))}")

    def test_reference_caps_describe_the_eager_path(self):
        """参考实现绝不能声称支持图捕获或自动调优：模型代码会去分配根本不存在的捕获缓冲区。"""
        caps = backends.reference.caps()
        self.assertEqual(caps.name, "reference")
        self.assertEqual(caps.device, "cpu")
        self.assertEqual(caps.native, frozenset())
        self.assertFalse(caps.graph)
        self.assertFalse(caps.autotune)
        self.assertFalse(caps.int8_weights)
        self.assertEqual(caps.attn_split, 1)
        self.assertTrue(caps.recurrent_state)


class RegistryTest(unittest.TestCase):
    """选择顺序与失败文案：显式钉住的后端绝不穿透，任何失败都不许无声。"""

    def test_available_names_include_every_discovered_vendor(self):
        """新增平台只是丢一个文件进去，所以可选名字必须来自发现过程。

        在这里钉一个字面量元组，等于把厂商目录本来要拆掉的那张中央清单又装回去：
        新的厂商文件在生产里可选，却会被这个测试拒绝。
        """
        names = backends.available_names()
        self.assertIn("hip", names)
        self.assertIn("reference", names)
        self.assertEqual(names[-1], "reference",
                         "the reference must sort last: it always answers usable")
        for module_name in backends.discovery.modules():
            vendor = module_name.rsplit(".", 1)[-1]
            self.assertIn(vendor, names,
                          f"{module_name} is a discovered vendor but not selectable")

    def test_reference_is_always_selectable(self):
        """它是兜底的最后一个候选；这条一旦失败，任何主机都跑不了引擎。"""
        self.assertIs(backends.select("reference"), backends.reference)
        self.assertTrue(backends.reference.probe())
        self.assertTrue(backends.reference.usable())
        self.assertEqual(backends.reference.reason(), "")

    def test_explicit_unknown_name_fails_with_the_known_names(self):
        """拼错名字不许落到自动探测：那会把部署失误藏在慢 10 倍的后端后面。"""
        with self.assertRaises(RuntimeError) as ctx:
            backends.select("no-such-backend")
        msg = str(ctx.exception)
        self.assertIn("no-such-backend", msg)
        for name in backends.available_names():
            self.assertIn(name, msg)

    def test_explicit_unusable_backend_fails_instead_of_falling_back(self):
        """钉住一个后端就是钉住它；悄悄换成别的跑，正是这个测试要拦的 bug。"""
        if backends.hip.usable():
            self.skipTest("this host has working HIP kernels; the unusable path needs a host without them")
        with self.assertRaises(RuntimeError) as ctx:
            backends.select("hip")
        msg = str(ctx.exception)
        self.assertIn("hip", msg)
        self.assertIn("reference", msg)

    def test_every_discovered_placeholder_is_registered_and_explains_itself(self):
        """能按名字解析出来的厂商文件，正是这个目录存在的意义。

        占位实现仍要回答 ``reason()``：运维刚插上的真实硬件上回一句「未知后端」，
        会让他们去查部署失误，而不是去查「这个集成还没做」。
        """
        for module_name in backends.discovery.modules():
            vendor = module_name.rsplit(".", 1)[-1]
            spec, backend = backends.discovery.load(module_name)
            self.assertEqual(spec.name, vendor)
            self.assertEqual(backend.name, vendor)
            self.assertFalse(backend.usable(),
                             f"{vendor} claims usable with no kernels")
            self.assertNotEqual(backend.reason(), "",
                                f"{vendor} is unusable but explains nothing")

    def test_hip_probe_rejects_a_cuda_namespace_that_is_not_rocm(self):
        """需要 ROCm 构建标记：光看命名空间认不出这张卡。

        好几家国产加速卡实现了 CUDA API（Iluvatar 的 CoreX 是有记录的案例：它自己的
        引擎直接选 ``cuda`` 设备），所以对任何可见的 CUDA 设备都答「可用」，会让这个
        后端认领一张它跑不了的卡，然后为该卡并不支持的 ISA 编译。

        这个测试*不*断言什么也值得写明：它不断言该标记充分。CoreX 是否也设置它尚无
        定论，这正是后端探测按认领强度排序的原因——Iluvatar 厂商按设备名匹配，无论
        如何都压过命名空间级的匹配。
        """
        if getattr(torch.version, "hip", None) is None:
            self.assertFalse(
                backends.hip.probe(),
                "probe() claimed a non-ROCm host: the build has no HIP marker, so any "
                "visible device belongs to another vendor")
            # ISA 是厂商判定的另一半：没有 ROCm 就没有 ISA。在这里编一个默认值，
            # 正是把错误的认领变成坏构建的那一步。
            self.assertEqual(backends.hip.caps().arch, "")

    def test_backend_is_memoised(self):
        """选两次会重新编译内核，或者让两个子系统对设备各执一词。"""
        self.assertIs(backends.backend(), backends.backend())
        self.assertIs(backends.backend().ops(), backends.backend().ops())

    def test_status_line_carries_name_device_native_graph_autotune(self):
        """banner 是静默回退到参考实现唯一能被看见的地方；缺一个字段就变得无法诊断。"""
        line = backends.status()
        for token in ("backend=", "device=", "native=", "graph=", "autotune="):
            self.assertIn(token, line)
        self.assertIn(f"native={len(backends.caps().native)}/{len(Ops.NAMES)}", line)

    def test_describe_all_covers_every_candidate(self):
        """诊断转储悄悄漏掉一个后端，「为什么选了那个」就再也说不清了。"""
        lines = backends.describe_all()
        names = backends.available_names()
        self.assertEqual(len(lines), len(names))
        for name, line in zip(names, lines):
            self.assertTrue(line.startswith(name + ":"), f"{name} -> {line!r}")


class ReferenceTotalityTest(unittest.TestCase):
    """参考后端必须答满 23 个名字：迁移完成后它是唯一的兜底。"""

    def test_reference_table_has_no_gaps(self):
        """这里留一个 ``None``，会在每个不支持的平台上以请求中途的形式撞进模型代码。"""
        table = backends.reference.ops()
        self.assertEqual(table.missing(), ())
        self.assertEqual(table.implemented(), Ops.NAMES)
        for name in Ops.NAMES:
            self.assertTrue(callable(getattr(table, name)), name)

    def test_every_vendor_table_resolves_to_reference_for_every_op(self):
        """占位实现今天就得答满 23 个名字：模型那边已经没有回退分支了。

        这条不变量让厂商可以从一个内核起步：它返回的部分表由参考实现补齐，硬件到位
        的第一天模型就能原样跑起来。
        """
        ref = backends.reference.ops()
        w = torch.randn(4, 3)
        x = torch.randn(3)
        for module_name in backends.discovery.modules():
            _spec, backend = backends.discovery.load(module_name)
            table = backend.ops()
            self.assertEqual(table.missing(), (), f"{backend.name} has gaps")
            for name in Ops.NAMES:
                self.assertTrue(callable(getattr(table, name)), f"{backend.name}.{name}")
            # 抽查补齐的确实是参考实现。
            self.assertTrue(torch.allclose(table.linear(w, x), ref.linear(w, x)),
                            f"{backend.name}.linear is not the reference op")

    def test_fill_keeps_native_entries_and_completes_the_rest(self):
        """解析器是「此处没有内核」变成「跑参考实现」的唯一一处；它坏了就处处都坏。"""
        marker = lambda *_a, **_k: "native"
        partial = Ops.from_mapping({"rms_norm": marker})
        filled = backends.reference.fill(partial, backends.reference.caps())
        self.assertIs(filled.rms_norm, marker)
        self.assertEqual(filled.missing(), ())
        self.assertEqual(set(filled.implemented()), set(Ops.NAMES))

    def test_hip_table_is_total_when_the_kernels_are_usable(self):
        """在 DCU 上每个条目都必须解析出来：HIP 表就是迁移之后生产上的参考实现。"""
        if not backends.hip.usable():
            self.skipTest("needs the HIP extension")
        table = backends.hip.ops()
        self.assertEqual(table.missing(), ())
        self.assertEqual(backends.caps().name, "hip")
        self.assertIn("linear", backends.caps().native)


class NativeRegistrationTest(unittest.TestCase):
    """后端的 ``_NATIVE`` 字典必须列出它为每个算子写的适配器。

    这里守的失效模式在其他任何检查里都看不见。``_NATIVE`` 是手工维护的
    name -> adapter 字典，从它里面漏掉的算子会被 ``reference.fill`` 用*参考实现*
    补上。于是适配器、头文件声明、绑定、协议字段、参考实现可以全都落地，整棵树
    也是绿的，那个算子却始终没在这个后端上跑过内核——它跑着 torch 参考实现，同时
    报告成功。在这种状态下测出来的性能会读成一个干净的零提升，与「融合没赚到」
    无法区分。

    检查针对适配器而不是内核符号：内核以自己的扩展层名字绑定（``gemv_kn``、
    ``silu_mul``），不必与算子名一致，所以这里只问一件事：模块为一个算子定义了
    适配器，却把它漏在注册之外。
    """

    def test_every_op_with_an_adapter_is_registered_native(self):
        """后端为某个算子定义了可调用的适配器，它就必须出现在 ``_NATIVE`` 里。"""
        from backends import corex

        missing = sorted(name for name in Ops.NAMES
                         if callable(getattr(corex, name, None))
                         and name not in corex.NATIVE)
        self.assertEqual(missing, [], (
            "these ops have an adapter in backends/corex but are absent from _NATIVE, so "
            "they silently resolve to the reference implementation and never run on the "
            f"card: {missing}"))

    def test_decode_op_set_is_a_subset_of_native(self):
        """``_DECODE_OPS`` 把守 ``caps.graph``；里面混进一个没有内核的名字，就可能捕获到 cuBLAS。"""
        from backends.corex import _DECODE_OPS, NATIVE

        self.assertEqual(sorted(_DECODE_OPS - set(NATIVE)), [])

    def test_every_native_op_is_bound_in_the_extension(self):
        """写在 ``.cu`` 文件里的内核，在 ``bindings.cpp`` 注册之前都取不到。

        这是上面 ``_NATIVE`` 那条守卫同一个陷阱的另一半，而且已经咬过一次：
        ``delta_normed`` 有声明、有定义、有协议字段、有参考实现、有适配器、也有
        ``_NATIVE`` 条目，可在卡上 decode 仍然死在 ``AttributeError``，因为 pybind 的
        ``m.def`` 那行没写。参考补齐和本地测试套件都看不见这一点：参考实现在 CPU
        主机上就能作答，而绑定只有导入扩展时才被读取。

        之所以做源码扫描，是因为没有工具链就导不进这个扩展。它天然是近似的——只证明
        某个名字出现在一次 ``m.def`` 调用里，不证明签名对得上——但它让这个遗漏无法
        留在树里，而真正发生过的正是这个遗漏。
        """
        import re

        from backends import corex

        bound_path = corex.kernels.KERNEL_DIR / "bindings.cpp"
        if not bound_path.is_file():
            self.skipTest("corex bindings not present")
        bound = set(re.findall(r'm\.def\(\s*"([a-z0-9_]+)"', bound_path.read_text(encoding="utf-8")))

        # 融合工作新增的算子，后端层算子名都由同名的扩展条目提供；内核以别的名字绑定
        # 的算子（``linear`` -> ``gemv_kn``）按设计不在本检查范围内，所以这里只对扩展
        # 头文件里也有声明的名字下断言。
        header_path = corex.kernels.KERNEL_DIR / "corex_ops.h"
        declared = set(re.findall(r"^(?:torch::Tensor|void)\s+([a-z0-9_]+)\s*\(",
                                  header_path.read_text(encoding="utf-8"),
                                  flags=re.MULTILINE))
        native_and_declared = (set(corex.NATIVE) & declared) | (declared & bound)
        missing = sorted(native_and_declared - bound)
        self.assertEqual(missing, [], (
            "these kernels are declared and listed native but never registered in "
            f"bindings.cpp, so calling them raises AttributeError on the card: {missing}"))


class VendorExtensionTest(unittest.TestCase):
    """新增平台必须只是丢一个文件进去，别的都不用改。

    整个层就是按这条性质来评判的：最早两个厂商是靠改三处中央清单接进来的，而平台层
    随厂商增多而腐化，走的正是这条路。
    """

    def test_a_vendor_file_is_a_complete_backend(self):
        """声明式路径必须只凭数据造出一个能用的后端。

        在仓库下的临时目录里造一个并把发现过程指过去，这样测的是真实的发现代码而不是
        桩。哪天这条要改注册表才能通过，就说明这个层退回成了中央清单。

        被测厂商是发现出来的，不是写死的：写死会把这条测试钉到 ``iluvatar`` 上，而那个
        平台拿到真内核、变成手写后端之后，就不再是声明了。「任意厂商文件」这种性质不
        应该因为某个厂商毕业而失效。
        """
        module_names = backends.discovery.modules()
        self.assertTrue(module_names, "no vendor declarations found to test with")
        spec, backend = backends.discovery.load(module_names[0])
        self.assertIsInstance(spec, backends.VendorSpec)
        self.assertIsInstance(backend, backends.SpecBackend)
        # 声明就是完整的后端：协议成员齐全，没有胶水代码。
        for member in ("probe", "usable", "reason", "ops", "caps", "device"):
            self.assertTrue(hasattr(backend, member), member)
        self.assertEqual(backend.ops().missing(), ())
        self.assertIn(backend.name, backends.available_names())

    def test_detection_yields_a_claim_strength_not_a_boolean(self):
        """认领强度是让共享命名空间能被正确解析的东西。

        Iluvatar 和 Hygon DCU 一样通过 CUDA 命名空间作答，所以两者都能看见设备。按名字
        匹配必须压过只按命名空间匹配，否则 ROCm 后端会抢走 Iluvatar 的卡。
        """
        self.assertGreater(backends.vendor.CLAIM_NAME, backends.vendor.CLAIM_NAMESPACE)
        self.assertGreater(backends.vendor.CLAIM_NAMESPACE, backends.vendor.CLAIM_NONE)

    def test_candidate_ranking_prefers_the_strongest_claim(self):
        """排序必须是全序且由认领驱动，绝不能落到文件系统顺序。"""
        strong = backends.registry.Candidate("a", backends.reference, claim=2, priority=90)
        weak = backends.registry.Candidate("b", backends.reference, claim=1, priority=10)
        self.assertLess(strong.rank(), weak.rank(),
                        "a name match must beat a namespace match regardless of priority")
        tie_a = backends.registry.Candidate("a", backends.reference, claim=1, priority=5)
        tie_b = backends.registry.Candidate("b", backends.reference, claim=1, priority=5)
        self.assertLess(tie_a.rank(), tie_b.rank(),
                        "an exact tie must still order deterministically, by name")

    def test_a_broken_vendor_file_is_reported_not_silently_dropped(self):
        """导入失败的厂商必须出现在诊断转储里。

        这里守的失效：有人加了一个平台，它的模块一导入就抛，而唯一的症状是
        ``COFFIN_BACKEND=<name>`` 回一句「未知后端」。
        """
        lines = backends.describe_all()
        joined = "\n".join(lines)
        for module_name in backends.discovery.modules():
            vendor = module_name.rsplit(".", 1)[-1]
            self.assertIn(vendor, joined, f"{vendor} missing from describe_all")

    def test_spec_rejects_a_declaration_that_cannot_work(self):
        """校验在发现时跑，不拖到没主硬件上的第一个请求。"""
        good = dict(name="x", detection=backends.Detection(namespace="cuda"))
        backends.VendorSpec(**good).validate()
        with self.assertRaises(ValueError):
            backends.VendorSpec(name="", detection=backends.Detection(namespace="cuda")).validate()
        with self.assertRaises(ValueError):
            backends.VendorSpec(**good, has_kernels=True).validate()
        with self.assertRaises(ValueError):
            backends.VendorSpec(**good, weight_layout="nx").validate()
        with self.assertRaises(ValueError):
            backends.VendorSpec(**good, attn_split=0).validate()


class ReferenceOracleTest(unittest.TestCase):
    """每个参考算子都必须复现本文件独立写出的那份数学。

    这些数字是新平台必须对齐的（PLATFORMS.md §6），所以对照实现是第二份实现，而不是
    调回后端。
    """

    def setUp(self):
        """每个测试固定种子，让失败仅凭测试名就能复现。"""
        torch.manual_seed(1234)
        self.ops = backends.reference.ops()

    def _linear_case(self, n=8, k=6):
        """权重与激活取 fp32，这是对照实现能精确比较的最宽 dtype。"""
        return torch.randn(n, k), torch.randn(k)

    def test_linear_single_row_and_batch_agree_with_a_matmul(self):
        """``linear`` 是解码热路径，也是模型唯一调用的投影；权重转置很像是对的，但答案错。"""
        w, x = self._linear_case()
        self.assertLess(_rel(self.ops.linear(w, x), x @ w.t()), FP32_TOL)
        X = torch.randn(5, w.shape[1])
        self.assertLess(_rel(self.ops.linear_m(w, X), X @ w.t()), FP32_TOL)
        self.assertLess(_rel(self.ops.linear_prefill(w, X), X @ w.t()), FP32_TOL)

    def test_linear_dispatches_on_int8_storage_not_on_an_op_name(self):
        """``QWeight`` 是模型侧的存储；int8 分支若留在调用点，按张量传参的平台就会坏掉。"""
        w, x = self._linear_case()
        q = QWeight.quantize(w)
        # 在 fp32 下反量化：``QWeight.dequant`` 默认 bf16，拿它比较量到的是 bf16 舍入
        # 而不是 int8 分派。
        dense = q.dequant(torch.float32)
        self.assertLess(_rel(self.ops.linear(q, x), x @ dense.t()), FP32_TOL)
        X = torch.randn(3, w.shape[1])
        self.assertLess(_rel(self.ops.linear_m(q, X), X @ dense.t()), FP32_TOL)
        self.assertLess(_rel(self.ops.linear_prefill(q, X), X @ dense.t()), FP32_TOL)

    def test_rms_norm_matches_the_definition_and_honours_a_row_pitch(self):
        """模型喂进的是融合缓冲区的列切片，所以假定行连续的变体会把 q_norm 写坏。"""
        x = torch.randn(6, 16)
        g = torch.randn(16)
        ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * g
        self.assertLess(_rel(self.ops.rms_norm(x, g, 16, 1e-6), ref), FP32_TOL)
        packed = torch.randn(6, 40)
        view = packed[:, :16]
        self.assertEqual(view.stride(0), 40)
        view_ref = view * torch.rsqrt(view.pow(2).mean(-1, keepdim=True) + 1e-6) * g
        self.assertLess(_rel(self.ops.rms_norm(view, g, view.stride(0), 1e-6), view_ref),
                        FP32_TOL)

    def test_rms_norm_gated_applies_silu_after_the_head_norm(self):
        """顺序有讲究：``silu(norm(x) * w)`` 与 ``norm(x) * w * silu(gate)`` 结果不同，checkpoint 里的是后者。"""
        x = torch.randn(4, 16)
        g = torch.randn(16)
        gate = torch.randn(4, 16)
        ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * g * F.silu(gate)
        self.assertLess(_rel(self.ops.rms_norm_gated(x, g, gate, 1e-6), ref), FP32_TOL)

    def _attn_inputs(self, nh=4, nkv=2, hd=8, window=6):
        """一段稠密的 int8 缓存窗口加一条查询，全用 fp32，比较才能精确。"""
        rep = nh // nkv
        kcache = torch.randint(-127, 127, (nkv, window, hd), dtype=torch.int8)
        vcache = torch.randint(-127, 127, (nkv, window, hd), dtype=torch.int8)
        kscale = torch.rand(nkv, window) + 0.1
        vscale = torch.rand(nkv, window) + 0.1
        return (torch.randn(nh, hd), torch.randn(nh, hd), kcache, vcache,
                kscale, vscale, rep, hd ** -0.5)

    def _attn_ref(self, q, gate, k, v, ks, vs, rep, scale, keep=None):
        """按定义直写的因果带门注意力：扩宽、缩放、掩码、softmax、PV、门控。"""
        nh, hd = q.shape
        nkv = k.shape[0]
        qe = q.reshape(nkv, rep, hd)
        w = k.float() * ks[:, :, None]
        scores = torch.einsum("ghd,gsd->ghs", qe, w) * scale
        if keep is not None:
            scores = scores.masked_fill(~keep[None, None, :], float("-inf"))
        probs = scores.softmax(-1)
        out = torch.einsum("ghs,gsd->ghd", probs, v.float() * vs[:, :, None])
        return out.reshape(nh, hd) * gate.sigmoid()

    def test_attn_decode_matches_the_reference_arithmetic(self):
        """注意力这里，缩放折了两次或掩码偏了一位，都还是看着合理但答案错。"""
        q, gate, kc, vc, ks, vs, rep, scale = self._attn_inputs()
        out = torch.zeros_like(q)
        self.ops.attn_decode(q, gate, kc, vc, ks, vs, out, torch.tensor([3]), rep, scale)
        ref = self._attn_ref(q, gate, kc[:, :4], vc[:, :4], ks[:, :4], vs[:, :4], rep, scale)
        self.assertLess(_rel(out, ref), FP32_TOL)

    def test_attn_decode_masks_past_the_position(self):
        """因果界是 ``pos``，不是缓存长度：整窗读完会悄悄泄漏未来的 token。"""
        q, gate, kc, vc, ks, vs, rep, scale = self._attn_inputs()
        short = torch.zeros_like(q)
        long = torch.zeros_like(q)
        self.ops.attn_decode(q, gate, kc, vc, ks, vs, short, torch.tensor([1]), rep, scale)
        self.ops.attn_decode(q, gate, kc, vc, ks, vs, long, torch.tensor([5]), rep, scale)
        self.assertGreater((short - long).abs().max().item(), 0.0)

    def test_attn_paged_equals_decode_on_an_identity_slot_map(self):
        """分页路径必须是同一个注意力；两者不一致，批处理与单流解码就会给出不同答案。"""
        q, gate, kc, vc, ks, vs, rep, scale = self._attn_inputs()
        window = kc.shape[1]
        pool_k = torch.zeros(12, kc.shape[0], kc.shape[2], dtype=torch.int8)
        pool_v = torch.zeros_like(pool_k)
        ps = torch.zeros(12, ks.shape[0])
        pv = torch.zeros_like(ps)
        pool_k[:window] = kc.permute(1, 0, 2)
        pool_v[:window] = vc.permute(1, 0, 2)
        ps[:window] = ks.t()
        pv[:window] = vs.t()
        dense = torch.zeros_like(q)
        paged = torch.zeros_like(q)
        pos = torch.tensor([3])
        self.ops.attn_decode(q, gate, kc, vc, ks, vs, dense, pos, rep, scale)
        self.ops.attn_paged(q, gate, pool_k, pool_v, ps, pv, torch.arange(12), paged,
                            pos, window, rep, scale)
        self.assertLess(_rel(paged, dense), FP32_TOL)

    def test_attn_paged_batch_equals_per_row_paged(self):
        """批内核的每一行都必须等于单行内核算那一行；否则换一个桶就换一套答案。"""
        nh, nkv, hd = 4, 2, 8
        rep = nh // nkv
        window, rows = 6, 3
        pool_k = torch.randint(-127, 127, (12, nkv, hd), dtype=torch.int8)
        pool_v = torch.randint(-127, 127, (12, nkv, hd), dtype=torch.int8)
        ps = torch.rand(12, nkv) + 0.1
        pv = torch.rand(12, nkv) + 0.1
        slots = torch.stack([torch.randperm(12) for _ in range(rows)])
        lens = torch.tensor([2, 4, 6], dtype=torch.int32)
        q = torch.randn(rows, nh, hd)
        gate = torch.randn(rows, nh, hd)
        got = torch.zeros(rows, nh, hd)
        self.ops.attn_paged_batch(q, gate, pool_k, pool_v, ps, pv, slots, lens, got,
                                  window, rep, hd ** -0.5)
        for b in range(rows):
            one = torch.zeros(nh, hd)
            self.ops.attn_paged(q[b], gate[b], pool_k, pool_v, ps, pv, slots[b], one,
                                torch.tensor([int(lens[b]) - 1]), window, rep,
                                hd ** -0.5)
            self.assertLess(_rel(got[b], one), FP32_TOL)

    def test_attn_paged_batch_uses_each_rows_own_length(self):
        """一次共享 launch 不能让某行的上下文长度改动另一行的答案；混合桶才是常态。"""
        nh, nkv, hd = 2, 2, 8
        rep = nh // nkv
        window = 6
        pool_k = torch.randint(-127, 127, (8, nkv, hd), dtype=torch.int8)
        pool_v = torch.randint(-127, 127, (8, nkv, hd), dtype=torch.int8)
        ps = torch.rand(8, nkv) + 0.1
        pv = torch.rand(8, nkv) + 0.1
        slots = torch.stack([torch.arange(8), torch.arange(8)])
        q = torch.randn(2, nh, hd)
        gate = torch.randn(2, nh, hd)
        mixed = torch.zeros(2, nh, hd)
        alone = torch.zeros(2, nh, hd)
        self.ops.attn_paged_batch(q, gate, pool_k, pool_v, ps, pv, slots,
                                  torch.tensor([1, 5], dtype=torch.int32), mixed,
                                  window, rep, hd ** -0.5)
        # 第 0 行的上下文两次都只有一个 key，所以第 1 行的上下文从五个 key 缩到
        # 一个时，它的输出不许动。
        self.ops.attn_paged_batch(q, gate, pool_k, pool_v, ps, pv, slots,
                                  torch.tensor([1, 1], dtype=torch.int32), alone,
                                  window, rep, hd ** -0.5)
        self.assertLess(_rel(mixed[0], alone[0]), FP32_TOL)
        self.assertGreater((mixed[1] - alone[1]).abs().max().item(), 0.0)

    def test_attn_paged_multi_q_equals_decoding_each_draft_alone(self):
        """校验只有在单 token 解码也同意时才接受草稿；一次共享的 pass 必须按查询逐位一致。"""
        nh, nkv, hd = 4, 2, 8
        rep = nh // nkv
        rows, ctx, c, window = 2, 3, 3, 8
        pool_k = torch.randint(-127, 127, (12, nkv, hd), dtype=torch.int8)
        pool_v = torch.randint(-127, 127, (12, nkv, hd), dtype=torch.int8)
        ps = torch.rand(12, nkv) + 0.1
        pv = torch.rand(12, nkv) + 0.1
        slots = torch.stack([torch.randperm(12), torch.randperm(12)])
        q = torch.randn(rows, c, nh, hd)
        gate = torch.randn(rows, c, nh, hd)
        got = torch.zeros(rows, c, nh, hd)
        self.ops.attn_paged_multi_q(q, gate, pool_k, pool_v, ps, pv, slots,
                                    torch.tensor([ctx] * rows, dtype=torch.int32),
                                    got, window, rep, hd ** -0.5)
        for b in range(rows):
            for j in range(c):
                one = torch.zeros(nh, hd)
                self.ops.attn_paged(q[b, j], gate[b, j], pool_k, pool_v, ps, pv,
                                    slots[b], one, torch.tensor([ctx + j]), window,
                                    rep, hd ** -0.5)
                self.assertLess(_rel(got[b, j], one), FP32_TOL)

    def test_attn_paged_multi_q_block_mask_excludes_blocks(self):
        """保留块掩码必须把被排除的 key 彻底丢掉；在 softmax 之后再掩分数做不到这一点。"""
        nh, nkv, hd = 2, 2, 8
        rows, ctx, c, window = 1, 1, 2, 8
        pool_k = torch.randint(-127, 127, (12, nkv, hd), dtype=torch.int8)
        pool_v = torch.randint(-127, 127, (12, nkv, hd), dtype=torch.int8)
        ps = torch.rand(12, nkv) + 0.1
        pv = torch.rand(12, nkv) + 0.1
        slots = torch.arange(12).reshape(1, 12)
        q = torch.randn(rows, c, nh, hd)
        gate = torch.randn(rows, c, nh, hd)
        got = torch.zeros(rows, c, nh, hd)
        # 块大小 2，只保留第一个块：查询 0 看到的是 2 个 key 的上下文里的 0..1，
        # 恰好就是不加掩码的情形。
        mask = torch.tensor([[1, 0, 0, 0]], dtype=torch.uint8)
        self.ops.attn_paged_multi_q(q, gate, pool_k, pool_v, ps, pv, slots,
                                    torch.tensor([ctx], dtype=torch.int32), got,
                                    window, nh // nkv, hd ** -0.5, None, None,
                                    mask, 2)
        for j in range(c):
            n = ctx + j + 1
            keep = torch.tensor([True, True] + [False] * j)
            ref = self._attn_ref(q[0, j], gate[0, j],
                                 pool_k[:n].permute(1, 0, 2),
                                 pool_v[:n].permute(1, 0, 2),
                                 ps[:n].t(), pv[:n].t(),
                                 nh // nkv, hd ** -0.5, keep)
            self.assertLess(_rel(got[0, j], ref), FP32_TOL)

    def test_conv1d_silu_shifts_its_window_and_matches_the_definition(self):
        """窗口平移是混合器的短期记忆：切片偏了会悄悄丢掉前三个 token。"""
        channels, kernel = 8, 4
        conv_w = torch.randn(channels, kernel)
        state = torch.randn(channels, kernel - 1)
        x = torch.randn(channels)
        state_in = state.clone()
        out = torch.zeros(channels)
        self.ops.conv1d_silu(conv_w, x, state, out)
        win = torch.cat([state_in, x[:, None]], dim=1)
        self.assertLess(_rel(out, F.silu((win * conv_w).sum(-1))), FP32_TOL)
        self.assertEqual(_rel(state, win[:, 1:]), 0.0)

    def test_conv1d_silu_seq_equals_step_by_step(self):
        """分块预填充不能在块的边界上重启那 3 个 token 的历史；状态存在的意义正是这个。"""
        channels, kernel = 8, 4
        conv_w = torch.randn(channels, kernel)
        x = torch.randn(9, channels)
        seq_state = torch.zeros(channels, kernel - 1)
        step_state = seq_state.clone()
        seq = self.ops.conv1d_silu_seq(conv_w, x, seq_state)
        steps = torch.stack([self.ops.conv1d_silu(conv_w, x[i], step_state,
                                                  torch.zeros(channels))
                             for i in range(x.shape[0])])
        self.assertLess(_rel(seq, steps), FP32_TOL)
        self.assertEqual(_rel(seq_state, step_state), 0.0)

    def test_conv1d_silu_batch_equals_per_row(self):
        """批处理不能把各行的卷积窗口混在一起；那会在序列之间交换 token。"""
        channels, kernel, rows = 8, 4, 3
        conv_w = torch.randn(channels, kernel)
        x = torch.randn(rows, channels)
        batch_state = torch.randn(rows, channels, kernel - 1)
        per_state = batch_state.clone()
        out = torch.zeros(rows, channels)
        self.ops.conv1d_silu_batch(conv_w, x, batch_state, out)
        for b in range(rows):
            one = torch.zeros(channels)
            self.ops.conv1d_silu(conv_w, x[b], per_state[b], one)
            self.assertEqual(_rel(out[b], one), 0.0)
        self.assertEqual(_rel(batch_state, per_state), 0.0)

    def _prologue_ref(self, q, k, v, b, a, neg_expA, dt_bias, qscale, rep):
        """按定义直写的头部准备：L2 归一化、头扩宽、sigmoid 与 softplus 门。"""
        qe = q.repeat_interleave(rep, dim=-2)
        ke = k.repeat_interleave(rep, dim=-2)
        qn = qe * torch.rsqrt((qe * qe).sum(-1, keepdim=True) + 1e-6) * qscale
        kn = ke * torch.rsqrt((ke * ke).sum(-1, keepdim=True) + 1e-6)
        return qn, kn, b.sigmoid(), neg_expA * F.softplus(a + dt_bias)

    def test_delta_prologue_matches_the_definition(self):
        """十六个小逐元素算子融合进了这一个；softplus 或头扩宽写错，递推会悄悄坏掉。"""
        nv, nk, dk, dv, rep = 4, 2, 8, 8, 2
        q, k = torch.randn(nk, dk), torch.randn(nk, dk)
        v, b, a = torch.randn(nv, dv), torch.randn(nv), torch.randn(nv)
        neg, dt = -torch.rand(nv) - 0.5, torch.randn(nv)
        qs = dk ** -0.5
        qo, ko, vo = torch.zeros(nv, dk), torch.zeros(nv, dk), torch.zeros(nv, dv)
        bo, go = torch.zeros(nv), torch.zeros(nv)
        self.ops.delta_prologue(q, k, v, b, a, neg, dt, qo, ko, vo, bo, go, qs, rep)
        qn, kn, bn, gn = self._prologue_ref(q, k, v, b, a, neg, dt, qs, rep)
        self.assertLess(_rel(qo, qn), FP32_TOL)
        self.assertLess(_rel(ko, kn), FP32_TOL)
        self.assertEqual(_rel(vo, v), 0.0)
        self.assertLess(_rel(bo, bn), FP32_TOL)
        self.assertLess(_rel(go, gn), FP32_TOL)
        self.assertLessEqual(go.max().item(), 0.0)

    def test_delta_prologue_batch_equals_per_row(self):
        """批处理头部准备把 ``neg_expA``/``dt_bias`` 在各行间共享；逐行读取这两个一维张量会混叠。"""
        nv, nk, dk, dv, rep, rows = 4, 2, 8, 8, 2, 3
        q, k = torch.randn(rows, nk, dk), torch.randn(rows, nk, dk)
        v, b, a = torch.randn(rows, nv, dv), torch.randn(rows, nv), torch.randn(rows, nv)
        neg, dt = -torch.rand(nv) - 0.5, torch.randn(nv)
        qs = dk ** -0.5
        qo = torch.zeros(rows, nv, dk)
        ko = torch.zeros(rows, nv, dk)
        vo = torch.zeros(rows, nv, dv)
        bo, go = torch.zeros(rows, nv), torch.zeros(rows, nv)
        self.ops.delta_prologue_batch(q, k, v, b, a, neg, dt, qo, ko, vo, bo, go, qs, rep)
        for i in range(rows):
            qn, kn, bn, gn = self._prologue_ref(q[i], k[i], v[i], b[i], a[i],
                                                neg, dt, qs, rep)
            self.assertLess(_rel(qo[i], qn), FP32_TOL)
            self.assertLess(_rel(ko[i], kn), FP32_TOL)
            self.assertLess(_rel(bo[i], bn), FP32_TOL)
            self.assertLess(_rel(go[i], gn), FP32_TOL)

    def test_delta_prep_seq_matches_the_definition_and_the_head_major_layout(self):
        """布局就是契约：分块扫描直接读 ``[H, T, d]``，输出转置了喂进去就是垃圾。"""
        nv, nk, dk, dv, rep, T = 4, 2, 8, 8, 2, 5
        conv_dim = 2 * nk * dk + nv * dv
        qkv, b, a = torch.randn(T, conv_dim), torch.randn(T, nv), torch.randn(T, nv)
        neg, dt = -torch.rand(nv) - 0.5, torch.randn(nv)
        qs = dk ** -0.5
        qt, kt = torch.zeros(nv, T, dk), torch.zeros(nv, T, dk)
        vt, bt, gt = torch.zeros(nv, T, dv), torch.zeros(nv, T), torch.zeros(nv, T)
        self.ops.delta_prep_seq(qkv, b, a, neg, dt, qt, kt, vt, bt, gt, qs, rep)
        kdim = nk * dk
        q, k = qkv[:, :kdim].reshape(T, nk, dk), qkv[:, kdim:2 * kdim].reshape(T, nk, dk)
        v = qkv[:, 2 * kdim:].reshape(T, nv, dv)
        qn, kn, bn, gn = self._prologue_ref(q, k, v, b, a, neg, dt, qs, rep)
        self.assertLess(_rel(qt, qn.permute(1, 0, 2)), FP32_TOL)
        self.assertLess(_rel(kt, kn.permute(1, 0, 2)), FP32_TOL)
        self.assertLess(_rel(vt, v.permute(1, 0, 2)), FP32_TOL)
        self.assertLess(_rel(bt, bn.permute(1, 0)), FP32_TOL)
        self.assertLess(_rel(gt, gn.permute(1, 0)), FP32_TOL)

    def test_delta_matches_the_gated_delta_update(self):
        """递推是 32 层里那 24 层唯一的跨 token 记忆：先衰减、再写、再读，顺序如此。"""
        nv, dk, dv = 4, 8, 8
        q, k, v = torch.randn(nv, dk), torch.randn(nv, dk), torch.randn(nv, dv)
        beta, g = torch.rand(nv), -torch.rand(nv)
        state = torch.randn(nv, dk, dv)
        got_st = state.clone()
        out = torch.zeros(nv, dv)
        self.ops.delta(q, k, v, beta, g, got_st, out)
        ref_st = state * g.exp()[:, None, None]
        kv = (ref_st * k[:, :, None]).sum(-2)
        d = (v - kv) * beta[:, None]
        ref_st = ref_st + k[:, :, None] * d[:, None, :]
        ref_out = (ref_st * q[:, :, None]).sum(-2)
        self.assertLess(_rel(out, ref_out), FP32_TOL)
        self.assertLess(_rel(got_st, ref_st), FP32_TOL)

    def test_delta_batch_equals_per_row_and_keeps_rows_independent(self):
        """各行不许共享状态：批处理步把更新广播出去，桶内成员就会互相影响。"""
        nv, dk, dv, rows = 4, 8, 8, 3
        q, k = torch.randn(rows, nv, dk), torch.randn(rows, nv, dk)
        v = torch.randn(rows, nv, dv)
        beta, g = torch.rand(rows, nv), -torch.rand(rows, nv)
        batch_st = torch.randn(rows, nv, dk, dv)
        per_st = batch_st.clone()
        out = torch.zeros(rows, nv, dv)
        self.ops.delta_batch(q, k, v, beta, g, batch_st, out)
        for b in range(rows):
            one = torch.zeros(nv, dv)
            self.ops.delta(q[b], k[b], v[b], beta[b], g[b], per_st[b], one)
            self.assertEqual(_rel(out[b], one), 0.0)
        self.assertEqual(_rel(batch_st, per_st), 0.0)


class DependencyDirectionTest(unittest.TestCase):
    """``backends/`` 不许导入它所服务的引擎层；单向导入才能让这一层单独可测。"""

    #: ``backends/`` 绝不能导入的包：它们在依赖图里位于这一层之上。
    FORBIDDEN = ("ops", "models", "engine", "serving", "cache", "entrypoints",
                 "engine_api")

    #: 唯一一处有记录的例外，连同移除它的步骤。S0 把主机侧 shim 与内核留在原处并把
    #: 它们适配成一个后端；S1 把它们移到 ``backends/hip/``，那时这条必须删除而不是
    #: 增长。``hip_ops`` 是*叶子* shim（只导入 torch），所以这份临时依赖不会把依赖图
    #: 反转。
    S0_ALLOWLIST = {"backends/hip/__init__.py": ("hip_ops",)}

    def _backend_files(self):
        """后端包里的每个 Python 文件，新的子模块自动被覆盖。"""
        root = ENGINE_DIR / "backends"
        return sorted(p for p in root.rglob("*.py"))

    def test_no_backend_module_imports_an_engine_layer(self):
        """这里出现一条导入就会反转依赖图，并让后端脱离 checkpoint 就无法测试。"""
        offenders = []
        for path in self._backend_files():
            rel = path.relative_to(ENGINE_DIR).as_posix()
            allowed = self.S0_ALLOWLIST.get(rel, ())
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                for name in names:
                    top = name.split(".")[0]
                    if top in allowed:
                        continue
                    if top in self.FORBIDDEN:
                        offenders.append(f"{rel}:{node.lineno} imports {name}")
        self.assertEqual(offenders, [],
                         "backends/ must not import engine layers:\n" + "\n".join(offenders))

    def test_the_s0_allowlist_stays_minimal(self):
        """反正则表越加越长，是依赖门禁沦为摆设的典型路径；这里把它钉在一条。"""
        self.assertEqual(set(self.S0_ALLOWLIST), {"backends/hip/__init__.py"})
        self.assertEqual(self.S0_ALLOWLIST["backends/hip/__init__.py"], ("hip_ops",))

    def test_package_imports_without_an_accelerator_or_a_build(self):
        """导入这个包就是纯 CPU 机器会做的事；在那里尝试构建会改写环境并失败。

        这里必须**显式**把 ``COFFIN_BACKEND=reference`` 传给子进程，不能依赖自动探测：
        ``_run_py`` 会清掉外层环境里的 ``COFFIN_BACKEND``，于是在**有卡**的主机上自动
        探测会正当地选中 ``corex``，这条断言就挂。它挂的原因不是代码坏了，是测试假设了
        「这台机器没有加速卡」—— 那正是它在本机能过、在卡上过不了的原因。
        """
        probe = ("import sys, backends\n"
                 "assert backends.caps().name == 'reference', backends.caps().name\n"
                 "assert 'hip_ops' not in sys.modules, 'imported the HIP shim'\n"
                 "print(backends.status())\n")
        done = _run_py(probe, env={"COFFIN_BACKEND": "reference"})
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("backend=reference", done.stdout)

    def test_env_var_selects_and_pins_the_backend(self):
        """``COFFIN_BACKEND`` 是做 A/B 与排障的开关；它必须在别的代码导入这个包之前就生效。"""
        ok = _run_py("import backends; print(backends.caps().name)",
                     env={"COFFIN_BACKEND": "reference"})
        self.assertEqual(ok.returncode, 0, ok.stderr)
        self.assertEqual(ok.stdout.strip(), "reference")

    def test_env_var_with_an_unknown_name_fails_loudly(self):
        """拼错的名字必须在导入时就中止，而不是选一个更慢的后端、看起来像性能回退。"""
        bad = _run_py("import backends", env={"COFFIN_BACKEND": "nonsense"})
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("nonsense", bad.stderr)
        self.assertIn("COFFIN_BACKEND", bad.stderr)


class CorexPackTest(unittest.TestCase):
    """int8 -> bf16 的权重量化转换，每次 int8 预填充调用都要付这个代价。"""

    def _packed(self):
        """造一个小尺寸、scale 接近真实的打包 int8 权重。

        Returns:
            一个 ``KnQWeight``，其 ``[K, N]`` 矩阵是连续的 int8，scale 是一维 fp32，
            正是构造函数校验的那一对。
        """
        from backends.corex.pack import KnQWeight

        k, n = 96, 40
        wt = torch.randint(-127, 127, (k, n), dtype=torch.int8)
        scale = (torch.rand(n) * 0.01 + 1e-4).to(torch.float32)
        return KnQWeight(wt, scale)

    def test_dequant_kn_is_bit_identical_to_the_wide_fp32_spelling(self):
        """快的写法不许挪动任何一个权重。

        原先的转换会先造一个 ``[K, N]`` 的 fp32 临时张量再窄化，在卡上每次预填充请求
        花 174.8 ms。先转成输出 dtype 再原地缩放快 1.70x，因为 ``mul_`` 的 dtype 取自
        目标张量，它仍在 fp32 下相乘、只舍入一次——同样的算术，没有临时张量。换成 bf16
        的 *scale* 反而会让每个权重差 1 ULP，这就是这里 scale 保持 fp32、这条测试要求
        逐位精确而不是近似的原因。
        """
        q = self._packed()
        got = q.dequant_kn(torch.bfloat16).wt
        want = (q.wt.float() * q.scale.view(1, -1)).to(torch.bfloat16)
        self.assertTrue(torch.equal(got, want),
                        "dequant_kn changed the weights; a faster conversion is only "
                        "acceptable if it is bit-identical")

    def test_dequant_kn_keeps_the_kn_layout(self):
        """结果直接喂给 ``matmul(lhs, wt)``；转置了或者不连续，乘法就会悄悄算错。"""
        q = self._packed()
        out = q.dequant_kn(torch.bfloat16)
        self.assertEqual((out.k, out.n), (q.k, q.n))
        self.assertEqual(tuple(out.wt.shape), (q.k, q.n))
        self.assertTrue(out.wt.is_contiguous())
        self.assertEqual(out.wt.dtype, torch.bfloat16)

    def test_dequant_kn_rejects_an_integer_destination(self):
        """往整型目标里缩放会回绕而不是抛错，事后读起来像一个损坏的 checkpoint。"""
        q = self._packed()
        with self.assertRaises(TypeError):
            q.dequant_kn(torch.int8)


class Int8BatchKernelTest(unittest.TestCase):
    """int8 批量投影必须是内核，不是看起来像内核的循环。

    这里守的失效在两个方向上都不出声。没有批量入口时，``linear_m`` 会循环调用单行
    GEMV，而那是*正确*的——功能测试看不见——但它每读一行就把整份权重读一遍，让每次
    投机校验都要多付 ``c`` 个普通步。这就是实测到投机后缀解码在这张卡上打不平的原
    因（../../PERF_COREX.md 第 14 节），而只要有人把分派改回循环，它立刻会回来。

    调内核需要卡，所以这些检查是结构性的：钉住必须同时存在的那几块，因为它们各自都
    在这个移植过程中当过缺失的那一块（../../PERF_COREX.md 12.5）。
    """

    def _sources(self) -> dict:
        """读出这份契约牵涉到的内核侧源码。

        Returns:
            ``{文件名: 文本}``，覆盖头文件、绑定和 GEMV 内核。

        Raises:
            unittest.SkipTest: 源码缺失，即一棵没有该后端的树。
        """
        from backends import corex

        kernel_dir = corex.kernels.KERNEL_DIR
        wanted = ("corex_ops.h", "bindings.cpp", "gemv.cu")
        if not all((kernel_dir / name).is_file() for name in wanted):
            self.skipTest("corex kernel sources not present")
        return {n: (kernel_dir / n).read_text(encoding="utf-8") for n in wanted}

    def test_the_batched_int8_entry_exists_in_header_binding_and_kernel(self):
        """声明、注册、定义是三处改动，必须全部落地。"""
        src = self._sources()
        self.assertIn("gemv_kn_i8_m", src["corex_ops.h"],
                      "the batched int8 entry is not declared in corex_ops.h")
        self.assertIn('m.def("gemv_kn_i8_m"', src["bindings.cpp"],
                      "the batched int8 entry is not registered in bindings.cpp, so "
                      "calling it raises AttributeError on the card instead of failing "
                      "the build")
        self.assertIn("gemv_kn_i8_m_kernel", src["gemv.cu"],
                      "gemv_kn_i8_m has no kernel body, so the entry has nothing to "
                      "launch")

    def test_the_batched_int8_kernel_accumulates_m_rows_per_weight_load(self):
        """只有 M 条累加链在一次权重读取期间一直活着，这个内核才划算。

        在源码里查，是因为这是编译器的产物，没有卡时任何单元测试都观察不到：行循环必须
        按 M 做模板实例化并完全展开，权重读取要提到循环外。用运行时的 ``for`` 遍历行，
        内核形状不变、权重却要读 M 次，那正是这个内核要消除的 bug。
        """
        src = self._sources()["gemv.cu"]
        start = src.index("gemv_kn_i8_m_kernel")
        body = src[start:src.index(chr(10) + "}" + chr(10), start)]
        self.assertIn("float acc[M * W];", body,
                      "the int8 batched kernel has no per-(row, column) accumulator array")
        self.assertIn("load_i8_row<W>(&wt[", body,
                      "the weight bytes are not loaded once into a local word, so they "
                      "are re-read once per row")
        self.assertIn("#pragma unroll", body,
                      "the row loop is not unrolled, so the M chains may spill or "
                      "re-read the weight")

    def test_linear_m_routes_int8_to_the_batched_kernel_not_a_row_loop(self):
        """分派正是「正确但更慢」的回退会悄悄改掉的那一块。"""
        import inspect

        from backends import corex

        source = inspect.getsource(corex._gemv_rows)
        self.assertIn("gemv_kn_i8_m", source,
                      "linear_m does not call the batched int8 entry: an int8 row "
                      "batch falls back to one weight read per row, the regression "
                      "that made speculative decoding unable to break even")


if __name__ == "__main__":
    unittest.main()
