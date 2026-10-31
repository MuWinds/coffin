"""运行时共享常量、设备运行时垫片和多模态校验辅助。

Module: ``coffin.serving``。本模块不持有任何引擎生命周期状态；它只放进程级默认
值、各个 mixin 共用的设备运行时访问器，以及运行时 mixin 使用的图像校验。

Constraints:
    * 不出现厂商拼写：设备运行时命名空间和每一处显存/计时调用都从
      ``backends.caps().device`` 推出来，绝不写字面量。正因如此，同一套服务代码
      才能在没有任何加速卡的机器上用 ``COFFIN_BACKEND=reference`` 起来——那里
      这个命名空间根本不存在。
    * import 本模块不许选定后端：命令行在应用 ``--backend`` 之前就从中读取
      ``MODEL`` / ``SOCK``，所以 ``backends`` 是在访问器内部 import 的，而不是
      放在模块作用域。
"""

from __future__ import annotations

import base64
import io
import os

import torch

MODEL = os.environ.get("ENGINE_MODEL", "/root/private_data/models/Qwen3.5-4B")
SOCK = os.environ.get("ENGINE_SOCK", "/tmp/qwen35.sock")
MAX_SEQ = 262144
MAX_DECODE_PIXELS = 16 * 1024 * 1024


def device_runtime():
    """返回选中后端对应的 torch 设备运行时命名空间。

    服务层每一处计时屏障和显存读数都需要同一个命名空间，而在这张卡上，承载厂商
    拼写的正是它。从 ``caps.device`` 推出来，就把那个拼写关在后端包里；没有可用
    设备时返回 ``None``，横幅和诊断代码才能在没有任何加速卡的主机上跑起来，而不
    是从一次没有保护的设备调用里抛出来。

    光有命名空间还不够判断。``torch.cpu`` 在每个构建里都存在，而且
    ``is_available()`` 返回 ``True``——它的意思是「CPU 一直在」，不是「挂着设备」
    ——而实现了 CUDA API 的厂商构建也会在没卡的机器上暴露设备模块。这两者都能
    通过 ``hasattr`` 测试，然后倒在第一次 ``synchronize`` 上，所以判断依据是后端
    自己的设备名加上一次可用性探测。

    Returns:
        ``torch.<caps.device>`` 模块；选中的后端在 CPU 上跑、torch 没有这个名字
        的命名空间、或者它报告没有可用设备时返回 ``None``。
    """
    import backends

    caps = backends.caps()
    if caps.device == "cpu":
        return None
    runtime = getattr(torch, caps.device, None)
    if runtime is None:
        return None
    probe = getattr(runtime, "is_available", None)
    if probe is None:
        return None
    try:
        return runtime if probe() else None
    # 驱动坏掉不能把启动横幅一起弄挂，所以探测失败按「没有运行时」处理。
    except Exception:  # noqa: BLE001
        return None


def sync() -> None:
    """阻塞到设备上排队的任务全部完成。

    计时窗口前后需要它：用主机侧时钟去量异步内核启动，量到的是启动，不是工作，
    ``bench`` 和 ``profile`` 会报出每步几微秒。CPU 后端既没有队列也没有命名空间，
    所以这里是空操作，而不是一个调用方可能忘记写的提前返回。
    """
    runtime = device_runtime()
    if runtime is not None:
        runtime.synchronize()


def empty_cache() -> None:
    """把缓存分配器的空闲段还给驱动。

    在拆解与下一次构建之间调用：池和 captured 的图是进程里最大的两笔分配，分配器
    否则会把它们的段留着，而那些显存正是重建要用的。
    """
    runtime = device_runtime()
    if runtime is not None:
        runtime.empty_cache()


def allocated_gib() -> float:
    """缓存分配器当前持有的活字节数，单位 GiB。

    Returns:
        所有分配器池的总和；后端没有分配器命名空间时是 ``0.0``。取整由调用方
        负责，因为这些读数都是给人盯显存预算看的，不用于比较。
    """
    runtime = device_runtime()
    if runtime is None:
        return 0.0
    return runtime.memory_allocated() / 2**30


def reserved_gib() -> float:
    """缓存分配器从驱动那里预留的字节数，单位 GiB。

    Returns:
        预留总量；后端没有分配器命名空间时是 ``0.0``。它总是和
        :func:`allocated_gib` 一起报，因为两者的差额就是重建时必须挤进去的碎片。
    """
    runtime = device_runtime()
    if runtime is None:
        return 0.0
    return runtime.memory_reserved() / 2**30


def free_bytes() -> int:
    """驱动报告的设备空闲显存，单位字节。

    Returns:
        空闲字节数；后端没有这个查询时返回 ``-1``，调用方因此可以对一个未知
        预算拒绝行动，而不是把它当成 0。
    """
    runtime = device_runtime()
    if runtime is None:
        return -1
    try:
        free, _total = runtime.mem_get_info()
    except Exception:  # noqa: BLE001 - a backend may expose the namespace without it
        return -1
    return int(free)


def profiler_activities(*, device: bool = False) -> list:
    """选中后端要用的 profiler 活动列表。

    Args:
        device: 是否连设备内核一起追踪。必须显式传，因为把解码步归因到某个 aten
            算子靠的是 CPU 那一半，只追设备的记录既没有启动名也没有主机时间。

    Returns:
        ``[CPU]``；``device`` 为真且后端确实是设备后端时是 ``[CPU, CUDA]``。
        这个枚举成员在每个 torch 构建里都存在，CPU 主机也会照单全收然后什么都不
        报——所以要问后端，而不是看符号在不在。
    """
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device and device_runtime() is not None:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return activities


def _eos_ids(tok) -> set[int]:
    """这个 checkpoint 上所有能停的 id，而不只是 ``eos_token_id``。

    对话模板用 ``<|im_end|>`` 结束 assistant 轮次，在这个 checkpoint 上它和模型的
    ``eos_token_id`` 是两个不同的 id。只按模型自己的 id 停，轮次结束后还会继续
    往下生成，把下一轮也吐出来。

    Args:
        tok: 已加载的分词器；哪些特殊 token 能解析出来由它的词表决定。

    Returns:
        停止 id 集合，一定包含 ``eos_token_id``。词表不认识的 token 会解析成负数
        id 并被丢掉，因为序列永远不可能发出它。
    """
    ids = {tok.eos_token_id}
    for t in ("<|im_end|>", "<|endoftext|>"):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0:
            ids.add(i)
    return ids


class MMError(Exception):
    """本服务拒绝的一个图像请求，同时带着要回的 HTTP 状态码。

    状态码跟着异常走，因为拒绝发生在传输层很下面的位置：各 op 处理器把它抄进
    回复，OpenAI 入口再把它映射成 HTTP 状态码，而不是把每次失败都算成 500。
    """

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = int(status)


def _decode_image(b64: str, mime: str, max_bytes: int):
    """把 ``data:`` URL 的载荷变成一张摆正、校验过的 RGB ``PIL.Image``。

    载荷解不开、超过 ``max_bytes``、或者超过 ``MAX_DECODE_PIXELS`` 时用
    ``MMError`` 拒绝（400 或 413，绝不会是 traceback）。PIL 自己的解压炸弹防护
    也算在内：容器没改 ``MAX_IMAGE_PIXELS``，仍是默认值，所以炸弹会在这里抛
    ``DecompressionBombError``，而不是把服务 OOM 掉。

    Args:
        b64: ``data:`` URL 的 base64 载荷，不含 ``data:`` 头。
        mime: 声明的媒体类型，只用在报错文本里。
        max_bytes: 这个请求允许的解码字节上限。

    Returns:
        ``(image, raw)``：解码后的 RGB 图像和它的原始字节。原始字节用于生成请求
        digest，给前缀缓存的键加盐，两张不同的图因此永远无法共用同一段缓存 KV。

    Raises:
        MMError: 载荷不是 base64（400）、解码为空（400）、超过 ``max_bytes``
            （413）、超过 ``MAX_DECODE_PIXELS``（413）、触发解压炸弹防护
            （413），或者不是能读的图像（400）。调用方原样转发状态码；重试没有
            帮助。
    """
    from PIL import Image, ImageOps, UnidentifiedImageError

    try:
        raw = base64.b64decode(b64, validate=True)
    except Exception as exc:  # noqa: BLE001 - any decode failure is the client's
        raise MMError(f"image payload is not valid base64: {exc}") from None
    if not raw:
        raise MMError("image payload decodes to zero bytes")
    if len(raw) > max_bytes:
        raise MMError(f"image is {len(raw):,} bytes decoded, over the "
                      f"{max_bytes:,}-byte (per-image) limit; downscale it first", 413)
    try:
        img = Image.open(io.BytesIO(raw))
        w, h = img.size
        if w * h > MAX_DECODE_PIXELS:
            raise MMError(
                f"image is {w}x{h} = {w * h / 1e6:.1f} MP; the decoder accepts at most "
                f"{MAX_DECODE_PIXELS / 1e6:.0f} MP (it is downsampled to the serving "
                f"size afterwards, but it has to be decoded first)", 413)
        # JPEG 可能是灰度或 CMYK，PNG 可能是调色板或 16 位；PIL 都能转。真正解码
        # 发生在 ``load()``，所以被截断的文件在这里就会带着可读的报错失败，而不是
        # 等到处理器内部才炸。
        img.load()

        # 手机照片把「旋转 90 度」存在 EXIF 里，模型读到一张横躺着的「SCNET」只会
        # 给出错误答案。
        img = ImageOps.exif_transpose(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
    except MMError:
        raise
    except Image.DecompressionBombError as exc:
        raise MMError(f"image is too large to decode safely: {exc}", 413) from None
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise MMError(f"image could not be decoded ({mime}): {exc}") from None
    return img, raw
