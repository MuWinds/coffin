"""测试套件共用的 fixture、跳过门与路径设置。

Module: ``coffin/tests``。每个测试模块都最先导入它，所以它不能反过来导入其中
任何一个；它也是唯一碰 ``sys.path``、唯一加载 checkpoint 的测试文件。

Constraints:
    * ``MKL_THREADING_LAYER=GNU`` 必须在 torch 初始化 MKL 之前导出，所以写在
      两条 import 之间，而不是放进 fixture。
    * 各测试模块按扁平名字导入 ``support``，因此都必须走 ``discover -s tests``
      这套布局，让 ``tests/`` 进入 ``sys.path``；``coffin/`` 再由本文件加进去，
      这样 ``ops`` / ``models`` / ``engine_api`` 才能按顶层包导入。
    * 同一时刻最多驻留一个模型：卡只有 16 GB，再加一份 4B 权重和两份 KV 缓存
      放不下，所以测试通过 fixture 申请配置，而不是自己另建一个模型。
    * 跳过门在导入时求值，需要 checkpoint 的测试模块必须在装饰类之前先导入
      ``support``。

套件在两个地方跑，跳过规则正是让两边都能用的原因：

* 没有 DCU 的开发机：内核与模型测试跳过，包结构测试照跑；
* SCNet 容器：全部跑在真实 HIP 内核与 ``$SCNET_MODEL`` 指向的 checkpoint 上
  （默认 ``/root/private_data/models/Qwen3.5-4B``）。

运行方式::

    cd /root/private_data/engine
    MKL_THREADING_LAYER=GNU PYTHONPATH=/root/private_data/engine \
        python3 -m unittest discover -s tests -t . -v
"""

from __future__ import annotations

import os

# 必须在 torch 初始化 MKL 之前设置：容器只带了 GNU 线程层，默认分派会以
# FATAL ERROR 直接中止（见 NOTES.md）。
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

import gc
import sys
import unittest
from pathlib import Path

import torch

#: ``coffin/`` 是所有包的 ``sys.path`` 根，``ops`` 与 ``models`` 才能是顶层包。
ENGINE_DIR = Path(__file__).resolve().parent.parent
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

#: 被测 checkpoint。``$SCNET_MODEL`` 优先，默认值是容器里存放它的位置。
MODEL_DIR = Path(os.environ.get("SCNET_MODEL")
                 or "/root/private_data/models/Qwen3.5-4B")

#: 随仓库提交的配置形状 fixture：让分派测试在没有 checkpoint（也没有可写的
#: 系统临时目录）时也能跑。
FIXTURE_MODEL_DIR = Path(__file__).resolve().parent / "fixtures" / "qwen3_5"


def has_cuda() -> bool:
    """是否有可见的 DCU；导入时据此构造跳过门。"""
    return torch.cuda.is_available()


def has_hip() -> bool:
    """HIP 扩展已构建且可加载时返回真（在纯 CPU 机器上永远不会构建）。

    判据是**选中的后端是不是 ``hip``**，而不是 ``import hip_ops``。那个顶层模块在
    CoreX 移植（``b05c677``）里搬进了 ``backends/hip/``，``import hip_ops`` 自此
    必然 ``ModuleNotFoundError``。

    这个坏掉的导入之所以长期没被发现，是因为它在 ``torch.cuda.is_available()`` 之后：
    没有卡的开发机上直接返回 ``False``，跳过门正常工作；**一旦在有卡的机器上跑全量
    ``unittest discover``，``support`` 在 import 阶段就炸，整个测试套件一条都跑不了。**
    也就是说这行代码只在真机上出错，而真机才是唯一能被它保护的地方。

    导入放在函数内部：选后端会加载已编译的扩展，在没有驱动的机器上会中止进程。
    """
    if not torch.cuda.is_available():
        return False
    import backends

    return backends.caps().name == "hip"


def has_model() -> bool:
    """checkpoint 是否在磁盘上；开销够小，可以在导入时调用。

    ``is_file()`` 会抛 ``PermissionError`` 而不是返回 ``False``：默认路径
    ``/root/private_data/models/Qwen3.5-4B`` 在集群上属于 root，而 ``SCNET_MODEL``
    没设时本模块在 import 阶段就访问它，于是**整个测试套件一条都跑不了**，报错还落在
    ``support`` 的 import 行上、看着像环境问题而不是「没设环境变量」。

    判据只用「读得到 config.json」，任何 ``OSError`` 都算没有：这条函数只决定跳不跳过，
    它不该有失败模式。
    """
    try:
        return (MODEL_DIR / "config.json").is_file()
    except OSError:
        return False


#: 需要 DCU。纯 torch 内核也要靠它持有张量。
needs_gpu = unittest.skipUnless(has_cuda(), "needs a DCU (torch.cuda unavailable)")
#: 需要手写的 HIP 内核（gemv / attn / delta / norm）。
needs_hip = unittest.skipUnless(has_hip(), "needs the HIP extension")
#: 需要磁盘上的 checkpoint（同时意味着需要 DCU）。
needs_model = unittest.skipUnless(
    has_cuda() and has_model(), f"needs a DCU and the checkpoint at {MODEL_DIR}")


def rel_err(got: torch.Tensor, ref: torch.Tensor) -> float:
    """相对 ``ref`` 的最大相对误差（与旧探针打印的是同一个度量）。

    用 ``ref`` 的最大幅值归一化，容差在任意层深下读起来都是同一个数；换成绝对
    容差就得逐层重新调。
    """
    return ((got - ref).abs().max() / ref.abs().max()).item()


class ModelFixture:
    """唯一驻留的那个模型，只有请求的配置变化时才重建。

    重建代价很高（加载权重、捕获计算图），所以 key 用完整的构造参数元组：配置
    相同的两个测试共用模型，改动一个旋钮的测试只付一次重载，而不是跑在一个为
    别人配置好的模型上。
    """

    _key: tuple | None = None
    _model = None
    _weights: dict | None = None

    @classmethod
    def get(cls, *, quant: bool = True, max_seq: int = 2048,
            paged: bool = False, bench: bool = False, weight_dtype=None):
        """返回共享模型；配置不同时先重载。

        Args:
            quant: 权重按 INT8（W8A16）存放，而不是 checkpoint 自带的 dtype。
            max_seq: KV 容量。超长的请求会被模型截断，所以探测长上下文的测试
                必须把它调大。
            paged: 启用分页 KV 缓存。分页路径与平坦路径的分配方式不同，不能共用
                同一个实例。
            bench: 分配基准测试形状的缓冲区。
            weight_dtype: 覆盖权重存放的 dtype（``torch.float16`` /
                ``torch.bfloat16``）；``None`` 沿用 checkpoint 自己的 dtype。

        Returns:
            活着的模型。它归这个类所有，不归调用方：不要关掉它，只在确实要拿回
            显存时才调 :meth:`close`。
        """
        key = (quant, max_seq, paged, bench, weight_dtype)
        if cls._key != key:
            cls.close()
            import engine_api
            kwargs = {} if weight_dtype is None else {"weight_dtype": weight_dtype}
            cls._weights = engine_api.load_weights(MODEL_DIR, quant=quant, **kwargs)
            cls._model = engine_api.create_model(
                MODEL_DIR, max_seq=max_seq, bench=bench,
                weights=cls._weights, paged=paged)
            cls._key = key
        return cls._model

    @classmethod
    def close(cls) -> None:
        """丢弃模型并交还显存，下次 :meth:`get` 会重新加载。

        在不同配置的模型之间省掉这一步，上一个模型会继续驻留，16 GB 的卡就被
        超订了。
        """
        if cls._model is not None:
            try:
                cls._model.close()
            finally:
                cls._model = None
        cls._weights = None
        cls._key = None
        gc.collect()
        if torch.cuda.is_available():
            # 捕获过的图会钉住一块私有内存池；只有图对象释放后池子才回到驱动
            # 手里，而这需要先同步再加一次回收，empty_cache 才真能释放这段。
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def ensure_captured(model):
    """把 decode 图捕获一次；``decode_step`` 需要一张可重放的图。"""
    if not model.graph_ready:
        model.capture()
    return model


class ModelTestCase(unittest.TestCase):
    """驱动已加载模型的测试的基类。

    子类设置 ``QUANT`` / ``MAX_SEQ`` / ``PAGED``；``setUp`` 取得共享 fixture 并
    把它倒回空上下文，让每个测试都从头开始。
    """

    QUANT = True
    MAX_SEQ = 2048
    PAGED = False
    WEIGHT_DTYPE = None

    @classmethod
    def setUpClass(cls) -> None:
        """以可读的理由跳过整个类，而不是在 ``setUp`` 里失败。"""
        if not (has_cuda() and has_model()):
            raise unittest.SkipTest(f"needs a DCU and the checkpoint at {MODEL_DIR}")

    def setUp(self) -> None:
        """取共享模型，并把它的上下文倒回空。"""
        self.model = ModelFixture.get(quant=self.QUANT, max_seq=self.MAX_SEQ,
                                      paged=self.PAGED,
                                      weight_dtype=self.WEIGHT_DTYPE)
        self.model.reset()

    def tearDown(self) -> None:
        """倒回上下文，免得下一个测试继承本次请求的状态。"""
        self.model.reset()
