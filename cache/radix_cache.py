"""分页 KV 块之上的基数树前缀缓存（RadixAttention）。

Module: ``coffin/``（顶层，不感知模型）。给 ``paged_cache`` 发出的块定键；
``block_manager`` 同时持有两者，也是调用方实际走的对象，而 ``prefix_engine``
（单流）和 ``model_runner``（批处理）是两条做匹配和插入的路径。本模块不许 import
模型族：它需要知道的关于模型族的一切，就是传进来的状态快照和 ``MMInput`` 上的
图像区间。

每一个见过的*按块对齐*的 token 前缀，都会连同持有它 KV 的物理块一起存进基数树。
新请求取最长的已缓存前缀（对那些块 ``refcount++``），只预填充后缀，于是共享的
系统提示只算一次，而不是每个请求算一次。

Constraints:
    * 只存整块、也只匹配整块。KV 就按块粒度共享，和 vLLM 的自动前缀缓存、SGLang
      的 RadixCache 一样；子节点用完整的第一个块做键，所以在第一个块内部就分叉的
      两个前缀仍然是两条不同的边。半块命中会让块表里留下一个不完整的块，而写时
      复制没法给它维护引用计数。
    * 在混合模型上只命中 KV 是静默错误的，所以匹配会被截断到带循环状态快照的最深
      边界（见下面的*混合模型*）。丢掉一个快照损失的是复用，绝不是正确性。
    * 块键带着该块的媒体签名（见下面的*媒体加盐*），而没有媒体信息时插入的边只
      允许被证明无图的数据块复用。这一步做错，就是把一张图的 KV 交给另一张图。
    * 不加锁。``llm_engine`` 的步进循环是唯一的写者，它轮流驱动两个调用方；
      ``refcount`` 数的是活着的序列数，不是块数。

混合模型：
Qwen3.5 把 8 层*全注意力*（分页 KV，可共享）和 24 层*线性注意力*交错排布，后者的
状态是整个前缀的函数。所以要复用一段 KV 前缀，必须把配套的循环状态也恢复回来
——光有 KV 等于把共享的 key 配上一个从没见过它们的状态。每个节点都带
``block_states``：沿该边每个块处线性注意力 conv/循环状态的可选快照，匹配因此
可以从任何一个有快照的块边界重放。``match_prefix(..., require_states=True)``
把匹配截断到带快照的最深边界，由调用方恢复那份状态。

快照很大（24 层 x [32,128,128] fp32 约 50 MB），所以有一个快照预算来限制：每个
存储前缀的末端节点有一个，中间的块边界则只在预算够的时候才有。没有快照就插入的
前缀仍然可以被匹配——调用方会丢掉没配到状态的块并重算，而绝不会从一个它恢复不了
的状态继续。

媒体加盐（图像）：
prompt 一旦能带图像，token id 就不再是全部故事：``<|image_pad|>`` 占位符段对
*每一张*图都一样，而为它算出的 KV 却取决于像素。只按 token id 定键的缓存会把
第一张图的 KV 交给第二张不同的图——静默的错误输出，prompt 越长错得越离谱。

所以每个块还带一个*媒体签名*：块内 token 的媒体 id 集合。被图 ``g`` 覆盖的
token 得到一个由该图内容 digest（``MMInput.digest``）和它在请求里的序号推出的
稳定整数，其他 token 一律取 :data:`TEXT_MEDIA_ID`（0）。文本块的签名是
``(0,)``，它的子键就是普通的 token 元组——纯文本路径于是保持了原来的键、原来的
行为和原来的开销：不传 ``mm`` 时整套签名计算完全跳过。图像块的键是
``(tokens, signature)``，两张图不同的 prompt 因此在第一个含图像 token 的块处
分叉。

只给图像块加盐就够了，因为基数树的路径就是*一条路径*：图像之后的块签名重新是
``(0,)``（里面没有图像 token），但唯一能走到它们的入口正是那个已经分叉的图像块。
这和 vLLM 把 mm 哈希并进前缀哈希是同一个论证，区别只是这里只有真正含 mm token 的
块才需要。id 还混入了图像在请求里的序号，同一张图出现两次，绝不会声称自己是另一
个位置上的同一份 KV。

有一处不对称，它决定加盐是真管用还是静默泄漏：*不带*媒体信息插入的边
（``mm=None``，也就是纯文本路径或任何早于此的调用方）没有签名，而「没有签名」
不能等于「谁都能复用」。这种边里的某个块可能是为一张没人记录过的图算出来的，所以
*确实*带媒体的查询只允许沿用它能证明无图的块。见 :meth:`RadixCache._sig_ok`。
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Optional

from .paged_cache import PagedKVCache
from utils.mm_input import IMAGE_TOKEN_ID

#: 一份线性注意力快照：模型层下标 -> (循环状态, conv 状态)。
LinearStates = dict[int, tuple[Any, Any]]

#: 不被任何图像覆盖的 token 的媒体 id（也是纯文本块的整个签名）。
TEXT_MEDIA_ID = 0


def media_id(digest: str, index: int) -> int:
    """为 ``digest`` 对应请求里的第 ``index`` 张图算一个稳定的正整数 int64。

    跨请求稳定（同样的字节 + 同样的位置得到同样的 id，对话的第二轮因此仍能命中），
    随图像内容不同而不同，而且永远不是 0（0 留给文本，这样纯文本块的签名才会
    保持 ``(0,)``）。

    Args:
        digest: 图像的内容 digest（``MMInput.digest``），两张不同的图因此不会
            共用 id。
        index: 这张图在请求图像中的序号。混入它，同一张图放在不同位置时就不会
            声称自己拥有同一份 KV。

    Returns:
        落在 int64 范围内的正整数 id：清掉最高位以保证不越界，最低位强制为 1
        以保证永不与 :data:`TEXT_MEDIA_ID` 冲突。
    """
    h = hashlib.sha256(f"{digest}#{index}".encode("utf-8")).digest()
    return (int.from_bytes(h[:8], "little") & ((1 << 63) - 1)) | 1


def media_ids_for(mm, num_tokens: int) -> Optional[list[int]]:
    """prompt 前 ``num_tokens`` 个 token 各自的媒体 id。

    返回 ``None`` 表示「没有媒体信息」：调用方是纯文本 prompt（或者压根没有
    ``MMInput``），树就继续用普通的 token 键。结果与 prompt 的*绝对*下标对齐，
    所以匹配 ``tokens[:-1]`` 的调用方传进来的仍是同一个列表的前缀。

    Args:
        mm: 请求的 ``mm_input.MMInput``，文本请求传 ``None``。只读 ``spans``
            和 ``digest``。
        num_tokens: 要为其生成 id 的前缀长度。spans 会被裁到这个长度，所以在
            ``tokens[:-1]`` 上做的匹配看到的 id，与在整个 prompt 上做的插入
            完全一致。

    Returns:
        按 prompt 顺序排列的每 token 一个媒体 id；prompt 完全不带媒体信息时为
        ``None``。

    Raises:
        ValueError: ``mm`` 有图像区间但 digest 为空。拒绝是安全的失败方式：没有
            内容 digest，两张尺寸相同的不同图会共用块键，缓存就会把一张图的
            KV 交给另一张。运行时服务总会把它填上
            （``mm_input.image_digest``），所以这是调用方的 bug。
    """
    if mm is None:
        return None
    spans = getattr(mm, "spans", None)
    if not spans:
        return None
    digest = getattr(mm, "digest", "") or ""
    if not digest:
        raise ValueError(
            "MMInput has image spans but an empty digest; the radix tree cannot "
            "salt its block keys (use mm_input.image_digest)")
    ids = [TEXT_MEDIA_ID] * num_tokens
    for index, (start, end) in enumerate(spans):
        mid = media_id(digest, index)
        a, b = max(int(start), 0), min(int(end), num_tokens)
        if a < b:
            ids[a:b] = [mid] * (b - a)
    return ids


def media_signature(media: Optional[list[int]], start: int, bs: int) -> tuple[int, ...]:
    """token ``start`` 处那个块的签名：排序后的媒体 id 集合。

    是集合而不是序列：块能否复用取决于它包含哪些图，而不是每张图在块内从哪开始，
    所以两个占位符填充方式不同的 prompt 仍然能命中同一个块。

    Args:
        media: :func:`media_ids_for` 给出的每 token 媒体 id；没有媒体信息的请求
            传 ``None``。
        start: 块起始的绝对 token。
        bs: 块的 token 数；签名覆盖 ``[start, start+bs)``。

    Returns:
        没有任何图像覆盖的块返回 ``(0,)``，否则返回它包含的、排好序的媒体 id。
    """
    if media is None:
        return (TEXT_MEDIA_ID,)
    ids = sorted(set(media[start:start + bs]))
    return tuple(ids) if ids else (TEXT_MEDIA_ID,)


class RadixNode:
    """树的一条边：它覆盖的 token，以及持有这些 KV 的块。

    节点的 ``key`` 是从父节点一直到它这里的整段 token，所以一条边可以覆盖多个块；
    ``blocks`` 与之平行，每块一项，``block_states`` 也一样（边里含图像时还有
    ``sigs``）。拆分一条边会保持这种平行结构。
    """

    __slots__ = ("parent", "children", "key", "blocks", "refcount",
                 "last_access", "block_states", "sigs")

    def __init__(self, parent: Optional["RadixNode"] = None, key: tuple = (),
                 blocks: Optional[list[int]] = None) -> None:
        """建一条边；由调用方把它挂进 ``parent.children``。

        Args:
            parent: 这条边挂靠的节点。只有根节点是 ``None``。
            key: 沿边的 token id。父节点用它的第一个块给子节点定键。
            blocks: 持有这些 KV 的物理块，与 ``key`` 的块一一对应。这里不取
                引用；那由插入的那个调用负责。
        """
        self.parent = parent
        self.children: dict[tuple, "RadixNode"] = {}
        #: 沿这条边的 token id；满足 ``len(key) == len(blocks) * bs``。
        self.key: tuple = tuple(key)
        self.blocks: list[int] = list(blocks or [])
        #: 停在这个节点上的活序列数；这是阻止一条边被淘汰的唯一东西。
        self.refcount = 0
        self.last_access = time.time()
        self.block_states: list[Optional[LinearStates]] = [None] * len(self.blocks)
        #: 每块的媒体签名。``None`` 表示这条边里没有图像 token，子键就是普通的
        #: token 元组，任何请求都能匹配它。
        self.sigs: Optional[list[tuple[int, ...]]] = None

    @property
    def is_leaf(self) -> bool:
        """这个节点之后是否不再有任何前缀。

        Returns:
            没有子节点时返回 ``True``；只有这种节点才可能被淘汰回收。
        """
        return not self.children

    def num_states(self) -> int:
        """这条边上有多少个块边界还留着快照。

        Returns:
            快照数；全树预算就是按它扣的。
        """
        return sum(1 for s in self.block_states if s is not None)


class RadixCache:
    """建在 :class:`~paged_cache.PagedKVCache` 的块之上的前缀缓存。"""

    def __init__(self, cache: PagedKVCache, max_state_snapshots: int = 8) -> None:
        """在 ``cache`` 的块之上建一棵空树。

        Args:
            cache: 块被共享的那个池。树对它存储的每个块都另取一份引用，所以这个
                池必须就是引擎分配时所走的那个对象。
            max_state_snapshots: 树最多保留多少份循环状态快照。一份约 50 MB
                （24 层的 [32,128,128] fp32），所以这是一笔显存预算；传 ``0``
                完全关掉快照（此时前缀复用只对纯文本或非混合模型有意义）。
        """
        self.cache = cache
        self.bs = cache.bs
        self.root = RadixNode()
        self.max_state_snapshots = max_state_snapshots
        self.hit_tokens = 0
        self.total_query_tokens = 0
        self.num_evicted_blocks = 0

    def match_prefix(self, tokens: list[int], require_states: bool = False,
                     mm=None) -> tuple[list[int], int, Optional[LinearStates]]:
        """``tokens`` 最长的已缓存前缀。

        返回 ``(blocks, matched_len, states)``。``matched_len`` 是 ``block_size``
        的整数倍，而且这些块的引用已经为调用方加好了：把它们收进一个
        :class:`~paged_cache.SeqKV`，用完后释放。

        ``mm``（一个 ``mm_input.MMInput``）会用每块包含的媒体给块键加盐；传
        ``None`` 就保持普通 token 键，文本请求不为加盐付任何代价。``tokens``
        是 prompt 的*绝对*前缀（调用方传的是 ``tokens[:-1]``），图像区间因此
        才对得上。

        Args:
            tokens: 要匹配的 prompt 前缀。短于一个块就完全不会命中；遍历只考虑
                整块。
            require_states: 把匹配截断到带循环状态快照的最深块边界，并释放它
                之后那些块的引用。混合模型上这是必须的，那里光有 KV 会把共享
                的 key 配上一个从没见过它们的状态。
            mm: 请求的图像载荷，纯文本传 ``None``。必须与之后交给 :meth:`insert`
                的是同一个对象。

        Returns:
            ``(blocks, matched_len, states)``：加了引用的块表、以 token 计的匹配
            长度（``0`` 且块表为空表示没命中），以及 ``matched_len`` 处的快照；
            ``require_states`` 关闭或该边界没有快照时为 ``None``。以
            ``matched_len`` 为准——``blocks`` 已经按它裁过。
        """
        self.total_query_tokens += len(tokens)
        bs = self.bs
        media = media_ids_for(mm, len(tokens))
        node, i = self.root, 0
        out: list[int] = []
        now = time.time()

        # 走过的路径上带快照的最深块边界；遍历结束后匹配要回退到这里，所以
        # 边走边记，而不是回头再翻树。
        cp_len, cp_n, cp_states = 0, 0, None

        while i + bs <= len(tokens):
            child = node.children.get(self._child_key(tokens, media, i))
            if child is None:
                break
            nb = 0
            m = len(child.blocks)
            while nb < m and (i + (nb + 1) * bs) <= len(tokens):
                seg = tuple(tokens[i + nb * bs:i + (nb + 1) * bs])
                if seg != child.key[nb * bs:(nb + 1) * bs]:
                    break
                if media is not None and not self._sig_ok(
                        media_signature(media, i + nb * bs, bs), child, nb, seg):
                    break
                nb += 1
            for b in child.blocks[:nb]:
                self.cache.incref(b)
                out.append(b)
            if require_states and nb > 0:
                # 从最新的边界往回想：最长的匹配前缀才值得留，那里的线性注意力
                # 状态才是调用方要恢复的。
                for k in range(nb - 1, -1, -1):
                    if child.block_states[k] is not None:
                        cp_len = i + (k + 1) * self.bs
                        cp_n = len(out) - (nb - (k + 1))
                        cp_states = child.block_states[k]
                        break
            i += nb * self.bs
            if nb < m:
                # 部分命中的边：前缀在这条边内部就分叉了，它下面不可能再匹配，
                # 遍历到此结束。
                break
            child.refcount += 1
            child.last_access = now
            node = child

        if require_states and cp_n < len(out):
            # 释放那些我们拿不出状态来交代的块：留着它们等于把一段循环状态无法
            # 恢复的 KV 交给调用方，而那正是快照存在要防的「状态静默不对」。
            # 重算这些块花的是时间，永远不花正确性。
            for b in out[cp_n:]:
                self.cache.decref(b)
            out = out[:cp_n]
            i = cp_len

        self.hit_tokens += i
        return out, i, cp_states

    def _sig_ok(self, sig: tuple[int, ...], node: "RadixNode", k: int,
                block_tokens: tuple) -> bool:
        """``node`` 这条边的第 ``k`` 块，可以被签名为 ``sig`` 的查询复用吗？

        三种情况，中间那种最微妙：

        * 边是*带*媒体信息插入的，有自己的签名：两边必须相等；
        * 边是*不带*媒体信息插入的（``sigs is None``：纯文本路径，或者早于
          ``mm`` 的调用方）。它的块完全可能是为一张我们一无所知的图算出来的，
          所以唯一能被证明安全的分支，是完全不含图像 token 的块。无条件接受它，
          正是本模块存在要防的那次静默跨图命中；
        * 加过盐的边存在加过盐的子键下，普通 token 的查找根本走不到它。

        Args:
            sig: 查询给这个块的签名。
            node: 正在考虑的边。
            k: 块在 ``node`` 这条边内的下标。
            block_tokens: 查询在这个块上的 token id，边不带签名时用来比对
                ``IMAGE_TOKEN_ID``。

        Returns:
            只有复用可证明时才返回 ``True``；任何未知情况都返回 ``False``，
            因为错答 ``True`` 的代价是错误输出，而不是一次缓存未命中。
        """
        if node.sigs is None:
            return sig == (TEXT_MEDIA_ID,) and IMAGE_TOKEN_ID not in block_tokens
        return sig == node.sigs[k]

    def _child_key(self, tokens: list[int], media: Optional[list[int]], start: int
                   ) -> tuple:
        """第一个块起于 ``start`` 的那个子边的键。

        Args:
            tokens: 正在匹配或插入的 prompt token。
            media: 每 token 的媒体 id，纯文本路径上是 ``None``。
            start: 子边第一个块的起始绝对 token。

        Returns:
            第一个块的 token 元组；只有该块含图像 token 时才在后面接上它的
            媒体签名。
        """
        key = tuple(tokens[start:start + self.bs])
        if media is None:
            return key
        sig = media_signature(media, start, self.bs)
        # 不含图像 token 的块一律保留*普通*键，无论请求是否带图：文本 prompt 和
        # 带图 prompt 因此能共享同一段文本前缀（纯文本的键格式也保持不变）。
        return key if sig == (TEXT_MEDIA_ID,) else key + sig

    def _edge_key(self, node: RadixNode) -> tuple:
        """``node.parent.children`` 用来存 ``node`` 的那个键。

        就是把 ``_child_key`` 的格式套到一个已有节点上，查找和当初建立该节点的
        链接因此永远一致；拆分时也用这个函数重新推出两半的键，而不是去记原来的
        键。

        Args:
            node: 正要挂到父节点下的节点。

        Returns:
            它第一个块的 token；该块含图像 token 时再接上它的签名。
        """
        key = tuple(node.key[:self.bs])
        if node.sigs is not None and node.sigs[0] != (TEXT_MEDIA_ID,):
            key += node.sigs[0]
        return key

    def insert(self, tokens: list[int], blocks: list[int],
               states_at: Optional[dict[int, LinearStates]] = None,
               mm=None) -> None:
        """插入一段按块对齐的前缀以及它的块。

        树会对 ``blocks`` 另取一份引用；调用方那份一直保留到自己的序列被释放为止。
        ``states_at`` 把绝对 token 位置（``block_size`` 的整数倍）映射到在那里
        观察到的线性注意力状态。``mm`` 给块键加盐的方式与 :meth:`match_prefix`
        完全一致（两边必须传同一个 ``MMInput``，否则插入会填进一条之后任何匹配都
        不会看的枝）。

        Args:
            tokens: 前缀覆盖的 token。没对齐块边界的尾巴在存入之前就被丢掉，
                因为那里永远不可能落下一个匹配。
            blocks: 持有这些 KV 的物理块，从位置 0 起，与 ``tokens`` 平行。树会
                给留下的部分加引用。
            states_at: 绝对位置 -> 在那里观察到的快照。缺的边界保持 ``None``，
                不能当命中点复用；已经有快照的位置不会被覆盖，也就是第一次观察
                的结果生效。
            mm: 用来给键加盐的媒体载荷；必须与之后调用 :meth:`match_prefix`
                时传的一致。
        """
        states_at = states_at or {}
        n_full = (len(tokens) // self.bs) * self.bs
        if n_full == 0:
            return
        tokens = list(tokens[:n_full])
        blocks = list(blocks[:n_full // self.bs])
        media = media_ids_for(mm, n_full)
        node, i = self.root, 0
        while i < len(tokens):
            child = node.children.get(self._child_key(tokens, media, i))
            if child is None:
                self._new_leaf(node, tokens[i:], blocks[i // self.bs:], i // self.bs,
                               states_at, media, i)
                return
            m = len(child.blocks)
            nb = 0
            while nb < m and (i + (nb + 1) * self.bs) <= len(tokens):
                seg = tuple(tokens[i + nb * self.bs:i + (nb + 1) * self.bs])
                if seg != child.key[nb * self.bs:(nb + 1) * self.bs]:
                    break
                if media is not None and not self._sig_ok(
                        media_signature(media, i + nb * self.bs, self.bs), child, nb,
                        seg):
                    break
                nb += 1
            if nb == m and (i + m * self.bs) < len(tokens):
                # 整条边吃完但还有 token 剩下：往下走。不动它的引用计数——插入
                # 只是存前缀，并不声称有序列停在这些前缀上。
                i += m * self.bs
                node = child
                continue
            if nb == m:
                # token 正好结束在子节点上：没有新东西要存，但调用方在这里拍到
                # 的快照仍可能补上空缺。
                self._adopt_states(child, i, nb, states_at)
                return
            mid = self._split(child, nb)
            rest_tok = tokens[i + nb * self.bs:]
            rest_blk = blocks[i // self.bs + nb:]
            if rest_tok:
                self._new_leaf(mid, rest_tok, rest_blk, (i + nb * self.bs) // self.bs,
                               states_at, media, i + nb * self.bs)
            return

    def _adopt_states(self, node: RadixNode, i: int, nb: int,
                      states_at: dict[int, LinearStates]) -> None:
        """给一条已有的边补上缺的快照，但不覆盖已有的。

        Args:
            node: 插入的 token 落在它的前缀上的那条边。
            i: 这条边起始的绝对 token 位置。
            nb: 本次插入覆盖到的该边块数。
            states_at: 绝对位置 -> 快照。节点已有的位置会跳过：一次迟到的重复
                观察不该替换掉活序列正在恢复的那份状态。
        """
        for k in range(nb):
            pos = i + (k + 1) * self.bs
            if node.block_states[k] is None and pos in states_at:
                node.block_states[k] = states_at[pos]
                self._trim_states()

    def _new_leaf(self, parent: RadixNode, key: list[int], blocks: list[int],
                  start_block: int, states_at: dict[int, LinearStates],
                  media: Optional[list[int]] = None, start_token: int = 0) -> RadixNode:
        """为一段树里还没有的前缀建一条边。

        Args:
            parent: 新边挂靠的节点。
            key: 沿边的 token，按块对齐。
            blocks: 持有这些 KV 的块；在这里加引用，因为树现在要持有一份。
            start_block: ``blocks[0]`` 在本次插入前缀里的下标，用来按绝对位置
                查尾巴的快照。
            states_at: 绝对位置 -> 快照。
            media: 本次插入前缀的每 token 媒体 id，或 ``None``。
            start_token: 这条边起始的绝对 token，签名因此从 ``media`` 的对应
                位置取起。

        Returns:
            新节点，已经挂到 ``parent`` 下。
        """
        node = RadixNode(parent=parent, key=key, blocks=blocks)
        for b in node.blocks:
            self.cache.incref(b)
        node.block_states = [states_at.get((start_block + k + 1) * self.bs)
                             for k in range(len(node.blocks))]
        node.sigs = self._signatures(media, start_token, len(node.blocks))
        parent.children[self._edge_key(node)] = node
        self._trim_states()
        return node

    def _signatures(self, media: Optional[list[int]], start_token: int,
                    n_blocks: int) -> Optional[list[tuple[int, ...]]]:
        """一条边的每块媒体签名；纯文本的边返回 ``None``。

        返回 ``None`` 是快路径：纯文本的边（里面没有图像 token）既保持原来的
        普通 token 行为，*也*能继续被完全不带 ``mm`` 的请求匹配。

        Args:
            media: 前缀的每 token 媒体 id，或 ``None``。
            start_token: 这条边起始的绝对 token。
            n_blocks: 沿边的块数。

        Returns:
            每块一个签名；没有任何块含图像 token 时返回 ``None``——正是这个表示
            让不带媒体信息的调用方也能匹配到这条边。
        """
        if media is None:
            return None
        sigs = [media_signature(media, start_token + k * self.bs, self.bs)
                for k in range(n_blocks)]
        return None if all(s == (TEXT_MEDIA_ID,) for s in sigs) else sigs

    def _split(self, child: RadixNode, nb: int) -> RadixNode:
        """在第 ``nb`` 块之后拆开 ``child``，返回新的上半节点。

        插入的 token 在 ``child`` 内部就分叉了，所以共享的头部独立成节点，剩下的
        尾巴挂在它下面。每个平行列表都在同一处切开，引用计数和 ``last_access``
        由头部继承——命中序列停靠的就是这个节点。

        Args:
            child: 正在拆的边。
            nb: 留在上半节点里的块数；至少 1，至多 ``len(child.blocks) - 1``。

        Returns:
            新的上半节点，已经挂到 ``child`` 原来的父节点下。
        """
        bs = self.bs
        mid = RadixNode(parent=child.parent, key=child.key[:nb * bs],
                        blocks=child.blocks[:nb])
        mid.refcount = child.refcount
        mid.last_access = child.last_access
        mid.block_states = child.block_states[:nb]
        mid.sigs = child.sigs[:nb] if child.sigs is not None else None
        child.parent.children[self._edge_key(mid)] = mid
        child.parent = mid
        child.key = child.key[nb * bs:]
        child.blocks = child.blocks[nb:]
        child.block_states = child.block_states[nb:]
        if child.sigs is not None:
            child.sigs = child.sigs[nb:]
            # 不含图像 token 的尾巴还原成 ``None`` 形式，纯文本请求因此仍能匹配
            # 到它。
            if all(s == (TEXT_MEDIA_ID,) for s in child.sigs):
                child.sigs = None
        mid.children[self._edge_key(child)] = child
        return mid

    def _trim_states(self) -> None:
        """丢掉超出预算的 LRU 快照（块本身保留）。

        每份快照约 50 MB，所以这个预算是一道显存上限，必须卡在插入时而不是读取时。
        丢掉一份降低的是之后匹配能达到的复用，而不是正确性：没有快照的边界不再是
        命中点。最久没被访问的先丢，因为对话是向前生长的，最老的边界也是最不可能
        再回去的地方。
        """
        if self.max_state_snapshots <= 0:
            return
        holders = []
        stack = [self.root]
        while stack:
            n = stack.pop()
            for k, s in enumerate(n.block_states):
                if s is not None:
                    holders.append((n.last_access, n, k))
            stack.extend(n.children.values())
        if len(holders) <= self.max_state_snapshots:
            return
        holders.sort(key=lambda h: h[0])
        for _, node, k in holders[:len(holders) - self.max_state_snapshots]:
            node.block_states[k] = None

    def _evictable_leaf(self) -> Optional[RadixNode]:
        """最久没被访问、且块没有任何其他持有者的叶子。

        叶子只有同时满足两个条件才可回收：没有序列停靠在它上面
        （``refcount == 0``），并且池里没有别的引用（``cache.refcount[b] == 1``，
        也就是只剩树自己那份）。淘汰一个仍被序列共享的块会破坏那条序列的 KV，所以
        这类叶子即使是叶子也会被跳过。

        Returns:
            淘汰候选；整棵树都被钉住时返回 ``None``。
        """
        best, best_t = None, float("inf")
        stack = [self.root]
        while stack:
            n = stack.pop()
            for c in n.children.values():
                if (c.is_leaf and c.refcount == 0
                        and all(self.cache.refcount[b] == 1 for b in c.blocks)):
                    if c.last_access < best_t:
                        best, best_t = c, c.last_access
                elif not c.is_leaf:
                    stack.append(c)
        return best

    def evict(self, blocks_needed: int) -> int:
        """按 LRU 淘汰无人共享的叶子，直到空出 ``blocks_needed`` 个块。

        Args:
            blocks_needed: 下一次分配想要的空闲块数。

        Returns:
            实际释放了多少块。少于请求数意味着树里已经没有可安全回收的东西了，
            调用方那次分配要么马上失败，要么得等某条序列退休——不能把这种情况
            当成成功。
        """
        freed = 0
        while self.cache.num_free < blocks_needed:
            leaf = self._evictable_leaf()
            if leaf is None:
                break
            for b in leaf.blocks:
                self.cache.decref(b)
            freed += len(leaf.blocks)
            self.num_evicted_blocks += len(leaf.blocks)
            del leaf.parent.children[self._edge_key(leaf)]
        return freed

    def reserve(self, blocks_needed: int) -> None:
        """为一次分配腾地方，必要时淘汰缓存前缀。

        只在分配之前调用，绝不之后：请求一旦自己握住了块，它本该让出的那些叶子也
        就不可淘汰了。

        Args:
            blocks_needed: 下一次预填充或解码步需要的空闲块数。
        """
        if self.cache.num_free < blocks_needed:
            self.evict(blocks_needed)

    def clear(self) -> None:
        """丢掉整棵树，释放每一个缓存块。

        只在没有序列持有从树里沿用的块时合法（引擎空闲时才调它）；
        ``LLMEngine.clear_prefix_cache`` 用它做冷启动的 A/B 测量。
        """
        stack = [self.root]
        while stack:
            node = stack.pop()
            for b in node.blocks:
                self.cache.decref(b)
            stack.extend(node.children.values())
        self.root = RadixNode()

    def num_cached_blocks(self) -> int:
        """树当前持有的块数，不论是否被共享。

        Returns:
            所有边上的块总数；这是池里在没有序列引用时能被淘汰回收的那部分。
        """
        n = 0
        stack = [self.root]
        while stack:
            node = stack.pop()
            n += len(node.blocks)
            stack.extend(node.children.values())
        return n

    def num_snapshots(self) -> int:
        """常驻的循环状态快照数，也就是预算限制的那个量。

        Returns:
            全树的快照数，每份约 50 MB 主机内存。
        """
        n = 0
        stack = [self.root]
        while stack:
            node = stack.pop()
            n += node.num_states()
            stack.extend(node.children.values())
        return n

    def snapshot_bytes(self) -> int:
        """常驻快照内存的主机侧估算（粗略）。

        每份快照按固定几何算：24 层线性注意力的 [32,128,128] fp32 循环状态，加上
        8192x3 的 fp32 conv 状态；这个常数是从 Qwen3.5 的结构推出来的，换一个
        模型族的混合模型就得换成自己的。它是给盯着 16 GB 预算的人看的，从不用于
        决定任何尺寸。

        Returns:
            估算字节数，精度只到快照数这一档。
        """
        return self.num_snapshots() * 24 * (32 * 128 * 128 * 4 + 8192 * 3 * 4)

    def depth(self) -> int:
        """存下来的最长前缀链长度，以边数为单位。

        Returns:
            从根到最深节点的边数：用来诊断复用实际能伸到多深，不构成任何上界。
        """
        best = 0
        stack = [(self.root, 0)]
        while stack:
            node, d = stack.pop()
            best = max(best, d)
            stack.extend((c, d + 1) for c in node.children.values())
        return best

    @property
    def hit_rate(self) -> float:
        """到目前为止从树里直接服务掉的查询 token 占比。

        Returns:
            ``hit_tokens / total_query_tokens``；第一次查询之前是 ``0.0``。
            这是个比例，只在一批负载上有意义，单个请求上没有。
        """
        if self.total_query_tokens == 0:
            return 0.0
        return self.hit_tokens / self.total_query_tokens
