"""PLE host 表的数据通路：分片映射、多分片寻址、两阶段预取与流水接线。

真实表是 128 个 ``[2500012, 160]`` 的 bf16 分片、共 102.4 GB，任何开发机都放不下，
所以这里在临时目录里造一张**同样布局**的小表：4 片 × 4224 行 × 4 宽 = 16896 行，
正好是这套小配置的 ``padded_vocab_size``。文件是真的 safetensors（手写 header 加
数据段），寻址用的整除/取模与真实表的 128 × 2500012 完全同构。

三条不变量，每条错了都不会抛异常、只会静默算错：

* **寻址**：行号 → (分片, 片内行)。差一片就是别人的行，而形状完全合法。
* **顺序**：按分片分组取行之后必须按原顺序写回，否则 token 与行错配。
* **两阶段**：``prefetch`` 与 ``wait`` 之间**有且只有一次**在飞。重复发、或者没发就
  取，都必须当场报错——中转缓冲只有一个，覆盖的时机对不对没法从数值上看出来。

跑法（仓库根目录）::

    cd coffin && python -m unittest tests.test_qwen4exp_ple_host -v

Module: ``coffin/tests``。依赖 ``models.qwen4exp`` 与 ``torch``；不碰 GPU、不读真实
checkpoint。``safetensors`` 只在集群上有（``checkpoint.py`` 的文件头就是这么写的），
所以本机读分片用 :func:`_open_with_torch` 顶上；装了那个包就自动换成
``checkpoint.open_shard_tensors``，见 :data:`OPENER`。
"""

from __future__ import annotations

import json
import mmap
import struct
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import torch

ENGINE = next((p for p in Path(__file__).resolve().parents
               if (p / "models" / "qwen4exp").is_dir()), Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ENGINE))

from models.qwen4exp.checkpoint import (  # noqa: E402
    MappedShards,
    ShardIndex,
    open_shard_tensors,
)
from models.qwen4exp.config import TextConfig  # noqa: E402
from models.qwen4exp.model import CausalLM, causal_mask  # noqa: E402
from models.qwen4exp.ngram import (  # noqa: E402
    HostNGramTable,
    NGramEmbedding,
    prefetch_ple_tables,
)
from models.qwen4exp.weights import NGRAM_SHARD_MARKER, load_host_ngram_table  # noqa: E402

#: safetensors 的 dtype 标签与 torch dtype 的对应。只列这套测试会写的那几种。
_ST_DTYPES = {torch.bfloat16: "BF16", torch.float32: "F32"}
_TORCH_DTYPES = {label: dtype for dtype, label in _ST_DTYPES.items()}

#: 造几个分片。真实表是 128 片，寻址代码对片数没有任何假设，4 片足够覆盖多组。
SHARD_COUNT = 4

try:  # pragma: no cover - 取决于机器上装没装
    import safetensors  # noqa: F401

    HAVE_SAFETENSORS = True
except ImportError:  # pragma: no cover
    HAVE_SAFETENSORS = False


def save_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    """手写一个 safetensors 文件（header 是 JSON，数据段紧跟其后）。

    为什么要自己写而不是 ``safetensors.torch.save_file``：本机没有那个包，而"从真的
    分片文件里按行取"这件事必须在本机验证得到。格式只有两段——8 字节小端 header
    长度、补齐到 8 字节的 JSON、然后是按 ``data_offsets`` 排好的数据。

    Args:
        path: 目标文件。
        tensors: 名字到张量的映射；张量按写入顺序依次进入数据段。
    """
    header: dict[str, dict] = {}
    blobs: list[bytes] = []
    offset = 0
    for name, tensor in tensors.items():
        tensor = tensor.contiguous()
        blob = tensor.view(torch.uint8).reshape(-1).numpy().tobytes()
        header[name] = {"dtype": _ST_DTYPES[tensor.dtype],
                        "shape": list(tensor.shape),
                        "data_offsets": [offset, offset + len(blob)]}
        blobs.append(blob)
        offset += len(blob)
    payload = json.dumps(header).encode("utf-8")
    # header 必须补齐到 8 字节：数据段起点 = 8 + len(header)，不齐会让读的人算错偏移。
    payload += b" " * ((-len(payload)) % 8)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(payload)))
        handle.write(payload)
        for blob in blobs:
            handle.write(blob)


def _header_of(path: Path) -> tuple[dict, int]:
    """读 safetensors 的 header（8 字节小端长度 + JSON），不碰数据段。

    Args:
        path: 分片文件。

    Returns:
        ``(header, header_len)``；数据段的绝对起点是 ``8 + header_len``。
    """
    with open(path, "rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        return json.loads(handle.read(header_len).decode("utf-8")), header_len


def shard_index(directory: Path) -> ShardIndex:
    """按目录里的分片 header 建一份索引。

    线上这份索引由 ``checkpoint.build_index`` 建（它要 ``safetensors``）；本机没有
    那个包，所以这里用同一批文件解析出来。``dtypes`` / ``shapes`` 两栏只有
    ``StreamingSource`` 用得到，装载 n-gram 表的路径只读 ``locations``。

    Args:
        directory: 含 ``*.safetensors`` 的目录。

    Returns:
        与 ``build_index`` 同形的索引。

    Raises:
        ValueError: 同一个张量名出现在两个分片里。
    """
    index = ShardIndex(shards=sorted(directory.glob("*.safetensors")))
    for path in index.shards:
        header, _ = _header_of(path)
        for name, info in header.items():
            if name in index.locations:
                raise ValueError(f"{name} 同时出现在 {index.locations[name]} 和 {path}")
            index.locations[name] = path
            index.dtypes[name] = info["dtype"]
            index.shapes[name] = list(info["shape"])
    return index


def _open_with_torch(names_and_paths) -> MappedShards:
    """用纯 torch 读 safetensors，返回与 :func:`checkpoint.open_shard_tensors` 同形的视图。

    开发机上没有 ``safetensors`` 包，而这套测试的价值恰恰在于**在本机**验证分片寻址
    与两阶段预取。这里读的是同一批真文件：解析 header、按 ``data_offsets`` 建 mmap
    视图，和 safetensors 的 ``get_slice(...).get_tensor()`` 做的是同一件事。

    Args:
        names_and_paths: ``(分片内张量名, 文件路径)``，按分片号升序。

    Returns:
        视图集合，以及保住映射的句柄（``MappedShards.close`` 会一起关掉）。

    Raises:
        KeyError: 名字不在那个文件的 header 里。
    """
    mapped = MappedShards()
    try:
        for name, path in names_and_paths:
            header, header_len = _header_of(path)
            info = header[name]
            begin, end = info["data_offsets"]
            handle = open(path, "rb")
            mapping = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
            mapped.handles.extend([mapping, handle])
            with warnings.catch_warnings():
                # torch 不支持"只读缓冲"，会为每个视图警告一次；这里就是要只读映射，
                # 写穿了也只会打在自己的临时文件上。
                warnings.simplefilter("ignore", UserWarning)
                raw = torch.frombuffer(mapping, dtype=torch.uint8, count=end - begin,
                                       offset=8 + header_len + begin)
            mapped.tensors.append(
                raw.view(_TORCH_DTYPES[info["dtype"]]).reshape(info["shape"]))
    except BaseException:
        mapped.close()
        raise
    return mapped


#: 有 safetensors 就跟线上走同一条读取器，没有就退回纯 torch 的那条。
OPENER = open_shard_tensors if HAVE_SAFETENSORS else _open_with_torch


def load_table(directory: Path, config: TextConfig):
    """按线上那条路径装载 host 表，只把 header 索引与读取器换成本机能跑的那两个。

    索引必须现建：有几个用例是先改目录内容（删片、改宽度）再装载的。

    Args:
        directory: 含 ``*.safetensors`` 的目录。
        config: 文本配置。

    Returns:
        :class:`~ngram.ShardedHostNGramTable`。
    """
    return load_host_ngram_table(directory, config, index=shard_index(directory),
                                 opener=OPENER)


def ple_config(**override) -> TextConfig:
    """与 ``test_qwen4exp_weights`` 同一套小配置。

    ``ple_embed_dim=64`` 配 16 个头，每行 4 个元素；``ngram_vocab_size_base=1000``
    推出来的 ``padded_vocab_size`` 是 16896，正好能被 4 整除。

    Args:
        **override: 要覆盖的字段。

    Returns:
        ``TextConfig``。
    """
    fields = dict(
        vocab_size=256, hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=32,
        linear_num_key_heads=2, linear_num_value_heads=6,
        linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
        num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32,
        shared_expert_intermediate_size=32, hc_count=4, hc_lowrank=16,
        ple_layer_ids=[2], ple_embed_dim=64, ple_conv_kernel_size=4,
        ngram_size=3, heads_per_ngram=8, ngram_vocab_size_base=1000,
        make_ngram_vocab_size_divisible_by=128,
        indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=16,
        indexer_budget=16, indexer_compress_ratio=4,
        output_gate_type="sigmoid", rope_theta=10000.0, partial_rotary_factor=0.25,
        mrope_section=[3, 3, 2], eos_token_id=2, pad_token_id=0,
    )
    fields.update(override)
    return TextConfig(**fields)


def ngram_geometry(config: TextConfig) -> tuple[int, int]:
    """这套配置下 PLE 表的 ``(总行数, 每行宽度)``。

    Args:
        config: 文本配置。

    Returns:
        ``(padded_vocab_size, head_dim_per_ngram)``。
    """
    embedding = NGramEmbedding(config, int(config.ple_embed_dim), 1, 0, stream=True)
    return embedding.padded_vocab_size, embedding.head_dim_per_ngram


def build_shard_dir(directory: Path, config: TextConfig, dtype: torch.dtype,
                    shard_count: int = SHARD_COUNT) -> torch.Tensor:
    """在 ``directory`` 里造一张分片 n-gram 表，返回**拼接后的参考表**。

    数据按固定种子生成，所以任何一个用例拿到的都是同一张表；返回的参考表就是
    「拼成一张大表」的那一份，各用例拿它做 ``index_select`` 对拍。

    Args:
        directory: 临时目录。
        config: 文本配置，决定行数与宽度。
        dtype: 表的 dtype；真实表是 bf16，整机前向要 fp32。
        shard_count: 分几片。

    Returns:
        ``[padded_vocab_size, head_dim]`` 的参考表。

    Raises:
        AssertionError: 行数不能被 ``shard_count`` 整除；整除映射要求每片一样多行。
    """
    rows, width = ngram_geometry(config)
    if rows % shard_count:
        raise AssertionError(f"{rows} 行装不进 {shard_count} 片")
    generator = torch.Generator().manual_seed(20260827)
    table = torch.randn(rows, width, generator=generator).to(dtype)
    per_shard = rows // shard_count
    for index in range(shard_count):
        name = ("model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
                f"{NGRAM_SHARD_MARKER}{index}.weight")
        save_safetensors(directory / f"model-{index:05d}-of-{shard_count:05d}.safetensors",
                         {name: table[index * per_shard:(index + 1) * per_shard]})
    return table


class _ShardCase(unittest.TestCase):
    """每个用例一套临时分片目录，结束时把表（连同文件句柄）一起放掉。

    ``close`` 必须显式调：Windows 上没关掉的映射会让临时目录删不掉，
    所以 ``TemporaryDirectory`` 也带上 ``ignore_cleanup_errors``。
    """

    dtype: torch.dtype = torch.bfloat16

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.dir = Path(self.tmp.name)
        self.config = ple_config()
        self.rows, self.width = ngram_geometry(self.config)
        self.full = build_shard_dir(self.dir, self.config, self.dtype)
        self.table = load_table(self.dir, self.config)

    def tearDown(self) -> None:
        self.table.close()
        self.tmp.cleanup()


class TestShardAddressing(_ShardCase):
    """行号 → (分片, 片内行) 的映射，与"拼成一张大表再 index_select"逐位相同。"""

    def test_geometry_matches_config(self):
        """分片几何必须等于模型算出来的几何。

        ``padded_vocab_size`` 是质数偏移的和再补齐，哈希出来的行号全都落在它里面；
        表的行数和它对不上时，越界与错片都会发生，而形状一路合法。
        """
        self.assertEqual(self.table.num_shards, SHARD_COUNT)
        self.assertEqual(self.table.rows_per_shard, self.rows // SHARD_COUNT)
        self.assertEqual(self.table.total_rows, self.rows)
        self.assertEqual(self.table.head_dim, self.width)
        self.assertEqual(self.table.dtype, self.dtype)

    def test_no_concatenated_copy(self):
        """分片表**不许**留一份拼好的 102.4 GB 连续副本。

        真实表 320001536 × 160 bf16 = 95.37 GiB；拼一份就多占一份匿名页、启动多走
        一遍 memcpy。这里用"所有分片的存储字节数之和恰好等于表的字节数"来钉住：
        多一份拼接副本会让这个和翻倍。
        """
        self.assertFalse(hasattr(self.table, "weight"),
                         "分片表不该再持有一张拼好的 weight")
        total_bytes = self.rows * self.width * torch.tensor([], dtype=self.dtype).element_size()
        mapped_bytes = sum(t.untyped_storage().nbytes() for t in self.table.shards)
        self.assertEqual(mapped_bytes, total_bytes)

    def test_gather_matches_concatenated_table(self):
        """随机行号的 ``gather`` 必须与 ``full.index_select`` 逐位相同。"""
        ids = torch.randint(0, self.rows, (6, 16), dtype=torch.long)
        got = self.table.gather(ids)
        expected = self.full.index_select(0, ids.reshape(-1)).reshape(6, -1)
        self.assertTrue(torch.equal(got, expected))

    def test_gather_across_shard_boundary(self):
        """跨片边界的那几行最容易写错：片内行号的取模、片号取整都在这里拐弯。

        边界取 ``每片行数`` 的整数倍及其两侧；错一位就会读到隔壁片的同一片内行，
        形状完全合法。
        """
        boundary = self.table.rows_per_shard
        probe = []
        for edge in range(1, SHARD_COUNT):
            probe.extend([edge * boundary - 1, edge * boundary, edge * boundary + 1])
        probe.extend([0, self.rows - 1])
        ids = torch.tensor(probe, dtype=torch.long)
        got = self.table.gather(ids)
        # 一维下标没有"头"那一维，返回的是 ``[N * head_dim]``（``ids.shape[:-1]`` 为空
        # 时的约定）；PLE 永远传 ``[B, S, ngram_heads]``。
        expected = self.full.index_select(0, ids).reshape(-1)
        self.assertTrue(torch.equal(got, expected))
        # 相邻两片的首尾行内容必须不同，否则上面的比较对小表是空断言。
        self.assertFalse(torch.equal(self.full[boundary - 1], self.full[boundary]))

    def test_gather_mixed_order_prefill(self):
        """乱序、跨片、几万行量级的一次取行（prefill 的形状）。

        这里同时压 ``[B, S, ngram_heads]`` → ``[B, S, ngram_heads * width]`` 的
        还原：写成按片号排序后就地返回，形状一样、内容整体错位。
        """
        ids = torch.randint(0, self.rows, (1, 512, 16), dtype=torch.long)
        got = self.table.gather(ids)
        self.assertEqual(tuple(got.shape), (1, 512, 16 * self.width))
        expected = self.full.index_select(0, ids.reshape(-1)).reshape(1, 512, -1)
        self.assertTrue(torch.equal(got, expected))

    def test_out_of_range_negative(self):
        """负行号必须抛 ValueError，且消息里带出实际范围。"""
        with self.assertRaises(ValueError) as caught:
            self.table.gather(torch.tensor([[0, -1, 3]], dtype=torch.long))
        message = str(caught.exception)
        self.assertIn("-1", message)
        self.assertIn(f"[0, {self.rows})", message)

    def test_out_of_range_high(self):
        """``>= 总行数`` 同样必须抛，且说明上界从哪来。"""
        with self.assertRaises(ValueError) as caught:
            self.table.gather(torch.tensor([[self.rows]], dtype=torch.long))
        self.assertIn(f"[0, {self.rows})", str(caught.exception))

    def test_rejects_uneven_shards(self):
        """分片行数不等必须当场拒绝：整除映射会整张错位，而形状完全合法。"""
        with self.assertRaises(ValueError):
            self.table.__class__(
                [torch.zeros(4, self.width), torch.zeros(5, self.width)], "cpu")

    def test_rejects_bad_shard_shape(self):
        """空列表与一维张量都不是表：前者后面每步都越界，后者连行都切不出。"""
        with self.assertRaises(ValueError):
            self.table.__class__([], "cpu")
        with self.assertRaises(ValueError):
            self.table.__class__([torch.zeros(4)], "cpu")


class TestTwoPhase(_ShardCase):
    """``prefetch`` / ``wait`` 的协议：有且只有一次在飞。"""

    def test_prefetch_wait_matches_gather(self):
        """``prefetch`` 之后 ``wait`` 到的行必须与 ``gather`` 逐位相同。"""
        ids = torch.randint(0, self.rows, (1, 24, 16), dtype=torch.long)
        self.table.prefetch(ids)
        self.assertTrue(self.table.has_pending())
        prefetched = self.table.wait()
        self.assertFalse(self.table.has_pending(), "wait 之后不该还有在飞的 prefetch，"
                                                  "否则下一次 prefetch 会炸")
        self.assertTrue(torch.equal(prefetched, self.table.gather(ids)),
                        "两阶段取到的行与同步取的不是同一批")

    def test_wait_without_prefetch_raises(self):
        """没发就取必须报错。静默返回上一次的行是这里最坏的失败方式。"""
        with self.assertRaises(RuntimeError):
            self.table.wait()

    def test_double_prefetch_raises(self):
        """连续两次 ``prefetch`` 必须报错：中转缓冲只有一个，第二次会覆盖第一次。"""
        ids = torch.zeros(4, 16, dtype=torch.long)
        self.table.prefetch(ids)
        with self.assertRaises(RuntimeError):
            self.table.prefetch(ids)
        # 第一次的结果仍然完好可取。
        self.assertTrue(torch.equal(self.table.wait(),
                                    self.full.index_select(0, ids.reshape(-1)).reshape(4, -1)))

    def test_take_falls_back_to_gather(self):
        """没有在飞时 ``take`` 退回同步取——单层直调（调试、fixture）走的就是这条。"""
        ids = torch.randint(0, self.rows, (2, 16), dtype=torch.long)
        self.assertFalse(self.table.has_pending())
        self.assertTrue(torch.equal(self.table.take(ids), self.table.gather(ids)))

    def test_take_consumes_pending(self):
        """有在飞时 ``take`` 消费它，且不再多发一次同步取。"""
        ids = torch.randint(0, self.rows, (2, 16), dtype=torch.long)
        self.table.prefetch(ids)
        self.assertTrue(torch.equal(self.table.take(ids), self.table.gather(ids)))
        self.assertFalse(self.table.has_pending())

    def test_take_rejects_shape_mismatch(self):
        """在飞的 prefetch 与本次下标形状不符时必须报错。

        两处下标不同源（换了层、换了 chunk）时取回来的是别人的行，形状却完全合法
        ——这是这套流水唯一会**静默**算错的地方。
        """
        self.table.prefetch(torch.zeros(3, 16, dtype=torch.long))
        with self.assertRaises(RuntimeError):
            self.table.take(torch.zeros(4, 16, dtype=torch.long))
        self.table.wait()

    def test_single_table_same_protocol(self):
        """整表版与分片版的两阶段行为必须一致——它们共用同一份中转逻辑。

        这条守的是重构：分片版是覆写 ``_fill_rows`` 加进来的，如果哪天两边的
        ``wait`` 语义分叉，真实路径（分片）和 fixture 路径（整表）就会各跑一套。
        """
        single = HostNGramTable(self.full, "cpu")
        try:
            ids = torch.randint(0, self.rows, (5, 16), dtype=torch.long)
            single.prefetch(ids)
            with self.assertRaises(RuntimeError):
                single.prefetch(ids)
            got = single.wait()
            with self.assertRaises(RuntimeError):
                single.wait()
            expected = self.full.index_select(0, ids.reshape(-1)).reshape(5, -1)
            self.assertTrue(torch.equal(got, expected))
            self.assertTrue(torch.equal(self.table.gather(ids), got))
        finally:
            single.close()


class TestLoadValidation(unittest.TestCase):
    """``load_host_ngram_table`` 的几何校验：分片对不上时必须在装载期失败。"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.dir = Path(self.tmp.name)
        self.config = ple_config()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_rejects_gap_in_shard_index(self):
        """分片号缺一片必须报错：行号的整除映射会整体前移，查到的全是别人的行。"""
        build_shard_dir(self.dir, self.config, torch.bfloat16)
        # 删掉 1 号片，剩下 0,2,3 —— 行数仍然"看起来"对，只有片号断档。
        (self.dir / "model-00001-of-00004.safetensors").unlink()
        with self.assertRaises(ValueError) as caught:
            load_table(self.dir, self.config)
        self.assertIn("连续", str(caught.exception))

    def test_rejects_row_count_mismatch(self):
        """总行数与配置不符必须报错，并同时报出两个数。"""
        rows, width = ngram_geometry(self.config)
        data = torch.zeros(rows // SHARD_COUNT, width, dtype=torch.bfloat16)
        for index in range(3):  # 3 片而不是 4 片：总数只有 3/4。
            name = ("model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
                    f"{NGRAM_SHARD_MARKER}{index}.weight")
            save_safetensors(self.dir / f"model-{index:05d}-of-00003.safetensors",
                             {name: data})
        with self.assertRaises(ValueError) as caught:
            load_table(self.dir, self.config)
        message = str(caught.exception)
        self.assertIn(str(rows), message)
        self.assertIn(str(data.shape[0] * 3), message)

    def test_rejects_width_mismatch(self):
        """每行宽度与模型不符必须报错。"""
        rows, width = ngram_geometry(self.config)
        data = torch.zeros(rows // SHARD_COUNT, width + 1, dtype=torch.bfloat16)
        for index in range(SHARD_COUNT):
            name = ("model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
                    f"{NGRAM_SHARD_MARKER}{index}.weight")
            save_safetensors(self.dir / f"model-{index:05d}-of-00004.safetensors",
                             {name: data})
        with self.assertRaises(ValueError):
            load_table(self.dir, self.config)


class TestPipelineWiring(_ShardCase):
    """``TextModel.forward`` 开头发 prefetch、PLE 层里 wait 的整条接线。

    表用 fp32：模型是 fp32，bf16 的表喂 bf16 才能进 ``Proj``，而这里要验的是接线
    不是 dtype 转换。
    """

    dtype = torch.float32

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = CausalLM(ple_config(), stream_ngram=True).eval()

    def _spy(self) -> dict:
        """记录这张表被 ``prefetch`` / ``wait`` 了几次，参数原样转发。"""
        calls = {"prefetch": [], "wait": 0}
        original_prefetch = self.table.prefetch
        original_wait = self.table.wait

        def prefetch(ids):
            calls["prefetch"].append(tuple(ids.shape))
            original_prefetch(ids)

        def wait():
            calls["wait"] += 1
            return original_wait()

        self.table.prefetch = prefetch
        self.table.wait = wait
        return calls

    def _run(self, table, ids: torch.Tensor, states=None, past: int = 0):
        """跑一次前向，返回 ``(logits, states)``。"""
        with torch.no_grad():
            return self.model(
                input_ids=ids,
                attention_mask=causal_mask(1, ids.shape[1], past),
                conv_mask=torch.ones(1, ids.shape[1], dtype=torch.bool),
                ple_tables={1: table}, states=states)

    def test_prefetch_once_then_wait_once(self):
        """一次前向只发一次 prefetch、只 wait 一次，结束后没有残留的在飞拷贝。"""
        calls = self._spy()
        ids = torch.randint(0, self.config.vocab_size, (1, 5))
        self._run(self.table, ids)
        self.assertEqual(len(calls["prefetch"]), 1,
                         "prefetch 必须在前向里发且只发一次")
        self.assertEqual(calls["prefetch"][0], (1, 5, 16),
                         "prefetch 的行集合就是 [B, S, ngram_heads]")
        self.assertEqual(calls["wait"], 1, "PLE 层必须消费那次 prefetch")
        self.assertFalse(self.table.has_pending(),
                         "前向结束后不该留下在飞的拷贝，否则下一步 prefetch 会报错")

    def test_matches_plain_table(self):
        """流水路径的 logits 必须与"直接传整张 host 张量"逐位相同。

        两条路径唯一的差别是行从哪来（预取还是同步查），任何一处寻址或顺序错位
        都会在这里显形。
        """
        ids = torch.randint(0, self.config.vocab_size, (1, 6))
        plain, _ = self._run(self.full, ids)
        piped, _ = self._run(self.table, ids)
        self.assertTrue(torch.equal(plain, piped), "流水路径与整表路径的 logits 不一致")

    def test_decode_step_uses_cached_context(self):
        """带状态的第二步同样逐位相同——prefetch 用的是缓存里的 n-gram 上下文。

        这一步专门守 ``prefetch_ple_tables`` 里取 ``states.layers[i].ple_context``
        的那几行：传 ``None`` 进去会让第二步对着"全是 eos 的假历史"哈希，形状全对、
        数值全错，而单步测试看不出来。
        """
        ids = torch.randint(0, self.config.vocab_size, (1, 5))
        plain_logits, plain_states = self._run(self.full, ids)
        piped_logits, piped_states = self._run(self.table, ids)
        self.assertTrue(torch.equal(plain_logits, piped_logits))

        step = torch.randint(0, self.config.vocab_size, (1, 1))
        plain_next, _ = self._run(self.full, step, states=plain_states, past=ids.shape[1])
        piped_next, _ = self._run(self.table, step, states=piped_states, past=ids.shape[1])
        self.assertTrue(torch.equal(plain_next, piped_next),
                        "解码步的 n-gram 上下文没有跟着状态走")

    def test_plain_tensors_are_not_prefetched(self):
        """普通 host 张量不该被这条流水拦下来——它没有两阶段协议。"""
        ids = torch.randint(0, self.config.vocab_size, (1, 3))
        started = prefetch_ple_tables(self.model.model.layers, ids,
                                      None, {1: self.full})
        self.assertEqual(started, 0)

    def test_layer_without_ple_is_rejected(self):
        """把 host 表配给没有 PLE 的层必须报错，而不是静默永不取用。"""
        ids = torch.randint(0, self.config.vocab_size, (1, 3))
        with self.assertRaises(ValueError):
            prefetch_ple_tables(self.model.model.layers, ids, None, {0: self.table})

    def test_same_table_on_two_layers_is_rejected(self):
        """同一张 host 表挂两层必须报错，而且**不能**留下没人取的 pending。

        真实 checkpoint 只有 ``layers.1`` 带 PLE；两层共用一张表时，第二层的行会覆盖
        第一层还在飞的那批，形状完全合法。校验要在发拷贝之前做完，否则报错之后再前向
        一次会在 prefetch 里再炸一次，错误信息还指向别处。
        """
        cfg = ple_config(num_hidden_layers=3, ple_layer_ids=[2, 3])
        model = CausalLM(cfg, stream_ngram=True).eval()
        ids = torch.randint(0, cfg.vocab_size, (1, 3))
        with self.assertRaises(ValueError):
            prefetch_ple_tables(model.model.layers, ids, None,
                                {1: self.table, 2: self.table})
        self.assertFalse(self.table.has_pending())


if __name__ == "__main__":
    unittest.main()
