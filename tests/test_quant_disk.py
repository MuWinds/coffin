"""预量化落盘的往返：写出去的那组字节读回来必须与运行时量化逐位相同。

Module: ``coffin/tests``。只用 ``torch`` 和 ``safetensors``，外加叶子模块
``int8_weights``、``quantized_load`` 和 ``tools``；不碰设备、不读 checkpoint，所以在
纯 CPU 主机上照跑。

这里守的不是「误差足够小」，而是 ``torch.equal``。离线预量化的唯一卖点是它和
「加载时量化」是同一次计算搬到了别的时刻；一旦两边差一个数，盘上的模型就不是线上
那个模型，而 logits 不会自己指出这一点——它只会给出一个略有不同的答案。

Constraints:
    * 断言一律 ``torch.equal``。容差会把「量化器改过了、盘上的权重没跟着更新」这类
      问题放过去，而那正是这个格式将来最可能的失效方式。
    * 临时根目录放在 ``coffin/.tmp_quant``（已 gitignore），不用 ``tempfile``：Windows
      上 mkdtemp 建的目录会带上沙箱令牌删不掉、也写不进去的 DACL，而 ``shutil.rmtree``
      对此无能为力，残留物会破坏所有人的工作区。
    * 后端必须被钉成声明 ``weight_layout="kn"`` 的那一个：加载器拒绝其它布局，而在
      纯 CPU 主机上默认选中的是 reference 后端（``nk``）。所以那个后端由
      ``unittest.mock.patch`` 提供一个只带 ``weight_layout`` 的替代品。
"""

from __future__ import annotations

import functools
import json
import shutil
import unittest
from pathlib import Path
from unittest import mock

import support

import torch

import quantized_load
from int8_weights import QWeight
from tools import quant_disk

#: 临时根目录。建在 ``.tmp_purity`` 下面，因为那正是纯度扫描器跳过、并且已 gitignore
#: 的目录：本模块造出来的 ``.safetensors``、``config.json`` 一旦被当成随代码发布的
#: 文件扫进去，就会拿它们里面的设备串把门禁弄红，而那是测试的残留，不是生产代码。
#: 用 ``tempfile`` 也不行——Windows 上 mkdtemp 建的目录带一个连删都删不掉的 DACL。
SCRATCH = Path(__file__).resolve().parents[1] / ".tmp_purity" / "quant"

try:  # 纯 CPU 主机上也可能没装 safetensors；那时整个模块跳过。
    import safetensors  # noqa: F401
    from safetensors.torch import save_file
    _HAS_SAFETENSORS = True
except Exception:                                   # pragma: no cover - env dependent
    _HAS_SAFETENSORS = False


def _scratch(name: str) -> Path:
    """在 :data:`SCRATCH` 下建一个干净的目录。

    Args:
        name: 目录名；同名残留会被先删掉，测试之间因此互不影响。

    Returns:
        建好的目录。
    """
    path = SCRATCH / name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def tearDownModule() -> None:
    """整轮跑完把 :data:`SCRATCH` 删掉。

    这是运行卫生，不是正确性要求：它已经在纯度扫描器跳过的目录里，残留不会弄红
    门禁。但一个每跑一次就多一份 ``.safetensors`` 的测试目录，迟早会被别人当成
    fixture。
    """
    shutil.rmtree(SCRATCH, ignore_errors=True)


def _kn_caps(weight_layout: str = "kn", name: str = "fake-kn"):
    """造一个只带 ``weight_layout`` 和 ``name`` 的 caps 替身。

    Args:
        weight_layout: 假后端声明的布局。
        name: 报错信息里显示的后端名。

    Returns:
        一个只暴露那两个字段的简单对象；加载器不读别的。
    """

    class _Caps:
        """加载器唯一会读的两个字段。"""

        weight_layout = "kn"
        name = "fake-kn"

    caps = _Caps()
    caps.weight_layout = weight_layout
    caps.name = name
    return caps


def _fake_backend(layout: str = "kn"):
    """把 ``backends.caps`` 换成返回指定布局的版本。"""
    module = mock.MagicMock()
    module.caps = functools.partial(_kn_caps, layout)
    return mock.patch.dict("sys.modules", {"backends": module})


@unittest.skipUnless(_HAS_SAFETENSORS, "needs the safetensors package")
class QuantizingTest(unittest.TestCase):
    """``quantize_projection`` 必须在分块与不分块两种情况下给出同一组字节。"""

    def test_chunking_does_not_change_a_single_byte(self):
        """分块是为了压住 fp32 临时量，不是为了换一种算法。

        行与行独立，所以每块的 absmax 只看自己那一行；一旦分块边界影响了结果，
        盘上的权重就取决于 ``_QUANT_CHUNK_ROWS`` 这个纯显存参数。
        """
        w = torch.randn(40, 24, dtype=torch.bfloat16)
        wt, scale = quant_disk.quantize_projection(w)
        # 逐行用运行时那条路的接口算一遍：这是「预量化 vs 运行时量化」的最小复现。
        want = QWeight.quantize(w)
        self.assertTrue(torch.equal(wt.t().contiguous(), want.wq))
        self.assertTrue(torch.equal(scale, want.scale))
        self.assertEqual(tuple(wt.shape), (24, 40))
        self.assertEqual(wt.dtype, torch.int8)
        self.assertEqual(scale.dtype, torch.float32)

    def test_multiple_full_chunks_stay_bit_identical(self):
        """跨过 ``_QUANT_CHUNK_ROWS`` 的边界之后仍然逐位相同。

        2 * chunk 行是唯一能同时覆盖「整块」和「尾块」的大小，而尾块正是边界上
        容易少算一行的地方。
        """
        rows = 2 * quant_disk._QUANT_CHUNK_ROWS + 7
        w = torch.randn(rows, 16, dtype=torch.bfloat16)
        wt, scale = quant_disk.quantize_projection(w)
        want = QWeight.quantize(w)
        self.assertTrue(torch.equal(wt.t().contiguous(), want.wq))
        self.assertTrue(torch.equal(scale, want.scale))

    def test_zero_rows_survive_the_floor(self):
        """全零行量化成零、反量化回零，而不是 NaN：``clamp_min(1e-8)`` 就是为此存在。"""
        w = torch.zeros(4, 8, dtype=torch.bfloat16)
        wt, scale = quant_disk.quantize_projection(w)
        self.assertTrue(torch.isfinite(scale).all())
        self.assertTrue(torch.equal(QWeight.from_kn(wt, scale).dequant(torch.float32),
                                    torch.zeros(4, 8)))

    def test_rejects_a_one_dimensional_tensor(self):
        """一维张量走到这里说明 ``is_quantizable`` 被绕过了，报错比转置它更有用。"""
        with self.assertRaises(ValueError):
            quant_disk.quantize_projection(torch.zeros(8, dtype=torch.bfloat16))


@unittest.skipUnless(_HAS_SAFETENSORS, "needs the safetensors package")
class RoundTripTest(unittest.TestCase):
    """写盘再读回来，``KnQWeight`` 拿到的必须是同一份 int8 和同一份 scale。"""

    def setUp(self) -> None:
        """造一个小 checkpoint：两个形状不同的投影、一个归一化权重、一个 ``A_log``。

        投影形状故意一个非方阵一个方阵：只有方阵看不出转错了方向，非方阵才能把
        ``[N, K]``/``[K, N]`` 弄混的错误变成一次形状断言。
        """
        self.root = _scratch("roundtrip")
        self.addCleanup(shutil.rmtree, self.root, True)
        self.ckpt = self.root / "ckpt"
        self.ckpt.mkdir()
        (self.ckpt / "config.json").write_text(
            json.dumps({"text_config": {"model_type": "qwen3_5_text"}}), encoding="utf-8")
        # 两个形状故意不同，而且一个是方阵：形状相同才看不出来转错了方向。
        self.tensors = {
            "model.language_model.layers.0.mlp.down_proj.weight":
                torch.randn(20, 12, dtype=torch.bfloat16),
            "model.language_model.layers.0.mlp.gate_proj.weight":
                torch.randn(12, 12, dtype=torch.bfloat16),
            "model.language_model.norm.weight": torch.randn(12, dtype=torch.bfloat16),
            "model.language_model.layers.0.linear_attn.A_log":
                torch.randn(8, dtype=torch.float32),
        }
        save_file(self.tensors, str(self.ckpt / "model-00001.safetensors"))
        self.out = self.root / "quant"

    def _convert(self, **kwargs) -> dict:
        """跑一遍转换，返回索引。"""
        return quant_disk.convert(
            self.ckpt, self.out, weight_dtype=torch.bfloat16, device=torch.device("cpu"),
            prefixes=quant_disk.DEFAULT_PREFIXES, include_vision=False,
            copy_sidecars=True, **kwargs)

    def test_the_written_shard_is_int8_kn_with_a_separate_scale(self):
        """落盘形状必须是 ``[K, N]`` 加另一个键的 fp32 ``[N]``。

        存成 ``[N, K]`` 会让每次加载都要转置，而转置正是这条路径要省掉的那一步。
        """
        from safetensors import safe_open

        index = self._convert()
        shard = next(iter(index["files"]))
        with safe_open(str(self.out / shard), "pt") as handle:
            keys = set(handle.keys())
            wt = handle.get_tensor("model.language_model.layers.0.mlp.down_proj.weight")
            scale = handle.get_tensor(
                "model.language_model.layers.0.mlp.down_proj.weight#scale")
        self.assertIn("model.language_model.layers.0.mlp.down_proj.weight#scale", keys)
        self.assertEqual(tuple(wt.shape), (12, 20))
        self.assertEqual(wt.dtype, torch.int8)
        self.assertEqual(tuple(scale.shape), (20,))
        self.assertEqual(scale.dtype, torch.float32)

    def test_loading_returns_byte_identical_weights(self):
        """读回来的 ``wq``/``scale`` 必须与运行时量化逐位相同（``torch.equal``）。"""
        self._convert()
        with _fake_backend():
            sd = quantized_load.load(self.out, device=torch.device("cpu"), quant=True)
        key = "model.language_model.layers.0.mlp.down_proj.weight"
        want = QWeight.quantize(self.tensors[key])
        got = sd[key]
        self.assertIsInstance(got, QWeight)
        self.assertTrue(torch.equal(got.wq, want.wq))
        self.assertTrue(torch.equal(got.scale, want.scale))

    def test_non_projections_keep_their_dtype(self):
        """归一化权重和 ``A_log`` 必须原样到达：消费它们的 bf16 内核没有 int8 通路。"""
        self._convert()
        with _fake_backend():
            sd = quantized_load.load(self.out, device=torch.device("cpu"), quant=True)
        norm = sd["model.language_model.norm.weight"]
        a_log = sd["model.language_model.layers.0.linear_attn.A_log"]
        self.assertEqual(norm.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(norm, self.tensors["model.language_model.norm.weight"]
                                    .to(torch.bfloat16)))
        self.assertEqual(a_log.dtype, torch.bfloat16)

    def test_conv1d_weight_is_not_quantized(self):
        """``conv1d.weight`` 被排除在 ``is_quantizable`` 之外，那条规则在这里也要生效。"""
        self.tensors["model.language_model.layers.0.linear_attn.conv1d.weight"] = (
            torch.randn(8, 1, 4, dtype=torch.bfloat16))
        save_file(self.tensors, str(self.ckpt / "model-00001.safetensors"))
        self._convert()
        with _fake_backend():
            sd = quantized_load.load(self.out, device=torch.device("cpu"), quant=True)
        conv = sd["model.language_model.layers.0.linear_attn.conv1d.weight"]
        self.assertNotIsInstance(conv, QWeight)
        self.assertEqual(conv.dtype, torch.bfloat16)

    def test_lm_head_is_read_when_present(self):
        """27B 是 untied 的，``lm_head`` 是最大的那个张量，也是唯一没有前缀的。

        漏掉它不会报错，头会悄悄退回用嵌入——那是错误答案而不是异常。
        """
        self.tensors["lm_head.weight"] = torch.randn(20, 12, dtype=torch.bfloat16)
        save_file(self.tensors, str(self.ckpt / "model-00001.safetensors"))
        self._convert()
        with _fake_backend():
            sd = quantized_load.load(self.out, device=torch.device("cpu"), quant=True)
        self.assertIn("lm_head.weight", sd)
        self.assertIsInstance(sd["lm_head.weight"], QWeight)

    def test_index_records_k_n_and_shapes(self):
        """索引是加载器校验形状的唯一依据，所以它记的 K/N 必须与落盘张量一致。"""
        index = self._convert()
        meta = index["tensors"]["model.language_model.layers.0.mlp.down_proj.weight"]
        self.assertEqual((meta["k"], meta["n"]), (12, 20))
        self.assertEqual(meta["kind"], "q8")
        self.assertEqual(index["weight_dtype"], "bfloat16")

    def test_config_json_is_copied_so_the_directory_is_a_checkpoint(self):
        """没有 ``config.json`` 时 ``engine_api`` 无法分派，目录只能当权重仓库用。"""
        index = self._convert()
        self.assertIn("config.json", index["sidecars"])
        self.assertTrue((self.out / "config.json").is_file())
        self.assertFalse((self.out / "model-00001.safetensors").exists())

    def test_source_checkpoint_is_left_alone(self):
        """checkpoint 一个字节都不许动；``--out`` 指到它自己时启动就该拒绝。"""
        before = (self.ckpt / "model-00001.safetensors").read_bytes()
        self._convert()
        self.assertEqual((self.ckpt / "model-00001.safetensors").read_bytes(), before)

    def test_a_tampered_config_is_rejected(self):
        """把另一份几何形状的配置配给这批权重，坏的是每一层的形状断言，离原因很远。"""
        self._convert()
        (self.out / "config.json").write_text('{"text_config": {}}', encoding="utf-8")
        with _fake_backend():
            with self.assertRaises(ValueError):
                quantized_load.load(self.out, device=torch.device("cpu"), quant=True)

    def test_a_tampered_shard_is_rejected(self):
        """索引说 ``[K, N]``、分片里是别的形状时必须报错，而不是错位地装配下去。"""
        index = self._convert()
        shard = next(iter(index["files"]))
        payload = {"model.language_model.layers.0.mlp.down_proj.weight":
                   torch.zeros(12, 19, dtype=torch.int8)}
        save_file(payload, str(self.out / shard))
        with _fake_backend():
            with self.assertRaises(ValueError):
                quantized_load.load(self.out, device=torch.device("cpu"), quant=True)

    def test_an_unknown_format_is_rejected(self):
        """布局一改，旧索引必须明确失败：按新布局错读出来的是一堆合理的数字。"""
        self._convert()
        path = self.out / quantized_load.INDEX_NAME
        index = json.loads(path.read_text(encoding="utf-8"))
        index["format"] = "coffin-int8-kn-v0"
        path.write_text(json.dumps(index), encoding="utf-8")
        with self.assertRaises(ValueError):
            quantized_load.read_index(self.out)

    def test_a_non_kn_backend_is_refused(self):
        """``nk`` 后端的内核会按连续内存索引一份非连续视图，读到错位字节且不报错。"""
        self._convert()
        with _fake_backend("nk"):
            with self.assertRaises(RuntimeError):
                quantized_load.load(self.out, device=torch.device("cpu"), quant=True)


class DetectionTest(unittest.TestCase):
    """``weights.load_weights`` 靠 ``is_quantized_dir`` 决定走哪条路。"""

    def setUp(self) -> None:
        self.root = _scratch("detect")
        self.addCleanup(shutil.rmtree, self.root, True)

    def test_an_empty_directory_is_not_a_quantized_checkpoint(self):
        """判据只有索引文件：别的判据都与「加载器需不需要它」无关。"""
        self.assertFalse(quantized_load.is_quantized_dir(self.root))

    def test_the_index_file_is_the_marker(self):
        """索引在场就算预量化目录，哪怕内容是空的——读它的人会立刻报错。"""
        (self.root / quantized_load.INDEX_NAME).write_text("{}", encoding="utf-8")
        self.assertTrue(quantized_load.is_quantized_dir(self.root))


class FromKnTest(unittest.TestCase):
    """``QWeight.from_kn`` 是这条路径不复制、不转置的全部依据。"""

    def test_from_kn_is_a_zero_copy_view(self):
        """``kn -> wq -> t().contiguous()`` 必须回到同一块内存，否则省掉的转置又回来了。"""
        wt = torch.randint(-127, 127, (5, 3), dtype=torch.int8)
        scale = torch.rand(3, dtype=torch.float32)
        q = QWeight.from_kn(wt, scale)
        self.assertEqual((q.n, q.k), (3, 5))
        self.assertEqual(q.wq.data_ptr(), wt.data_ptr())
        back = q.wq.t().contiguous()
        self.assertEqual(back.data_ptr(), wt.data_ptr())
        self.assertTrue(torch.equal(back, wt))

    def test_from_kn_dequantises_the_same_way_as_quantize(self):
        """同一个矩阵，两种包装方式反量化必须逐位相同。"""
        w = torch.randn(6, 4, dtype=torch.bfloat16)
        want = QWeight.quantize(w).dequant(torch.float32)
        q = QWeight.quantize(w)
        got = QWeight.from_kn(q.wq.t().contiguous(), q.scale).dequant(torch.float32)
        self.assertTrue(torch.equal(got, want))

    def test_from_kn_rejects_a_bad_pairing(self):
        """形状对不上时拿错元素算出来的数看着合理，所以在构造点拒绝。"""
        with self.assertRaises(ValueError):
            QWeight.from_kn(torch.zeros(5, 3, dtype=torch.int8), torch.zeros(4))
        with self.assertRaises(ValueError):
            QWeight.from_kn(torch.zeros(5, 3), torch.zeros(3))
        with self.assertRaises(ValueError):
            QWeight.from_kn(torch.zeros(5, 3, dtype=torch.int8),
                            torch.zeros(3, dtype=torch.bfloat16))


if __name__ == "__main__":
    unittest.main()
