"""主机侧的 suffix 草稿器：用请求自身的 token 历史做投机解码。

Module: ``coffin/ops``。引擎的投机步骤用它取代 MTP
草稿模型；``coffin`` 里没有任何东西向下调用本模块，所以它待在依赖图之外，
自己也不带任何依赖。

Constraints:
    * 纯 Python，只用标准库，不 import ``torch``，不碰设备，不起线程。本引擎
      针对的那张卡会被 CUDA 图搞坏（见 AGENTS.md），草稿模型因此不可用，投机
      只能来自 token 本身，且绝不能扰动设备。``ops`` 包的 ``__init__`` 为了
      内核而 import torch；这里不许再加码，否则这个草稿器就无法在纯 CPU 主机上
      测试了。
    * 索引只由 ``reset`` 重建、只由 ``observe`` 扩展，所以草稿里不可能出现
      模型还没产出的 token。历史来自目标模型自己的输出；一旦有 bug 把草稿喂回
      历史，索引就变成自证，接受率也失去意义。
    * 先试最长后缀，再看后续频率。目标模型会校验每一份草稿，所以一次错误的
      长匹配只浪费一个被拒槽位，而短匹配通常本来就够得着；优先长匹配才能提高
      每次校验的接受 token 数。
    * 一份草稿只是几次字典查找、零次权重读取：考虑到这颗芯片上每步解码
      20.26 ms，这正是本模块相对草稿模型能站得住脚的成本模型。
"""

from __future__ import annotations


class SuffixDrafter:
    """通过查找最近后缀在更早处出现的位置来预测接下来的 token。

    草稿器持有 token 历史，外加一个索引：从每个 m token 的窗口
    （``1 <= m <= max_match``）指向该窗口结束的位置。同一个后缀草稿两次就是一次
    哈希命中，权重一次都不读。

    两次 reset 之间索引只追加，``observe`` 的复杂度才是
    ``O(len(tokens) * max_match)``，而不是每步重建一遍。
    """

    def __init__(self, max_match: int = 4, min_match: int = 2) -> None:
        """设定一次查找可用的最长与最短后缀。

        Args:
            max_match: 最长后缀的 token 数。默认 4，因为更长的查找在一个 prompt
                里很少出现两次，而每次落空都是一次白探索引。
            min_match: 仍算作匹配的最短后缀。取 1 时任何位置都能命中，草稿出来的
                是纯噪声，还要每步占掉一个校验槽位。

        Raises:
            ValueError: ``min_match < 1`` 或 ``max_match < min_match``。前者让匹配
                循环为空，后者让它失去意义；而悄悄什么都草稿不出来，看起来会像
                质量问题而不是配置 bug。
        """
        if min_match < 1:
            raise ValueError(f"min_match must be >= 1, got {min_match}")
        if max_match < min_match:
            raise ValueError(
                f"max_match ({max_match}) must be >= min_match ({min_match})"
            )
        self.max_match = max_match
        self.min_match = min_match
        self._history: list[int] = []
        # 每个 m 一个字典，键是 m token 的窗口，值是升序的结束位置。按 m 拆开让
        # 键里不必再带长度，而且按 m 分开的字典比一个以 (m, window) 为键的大字典
        # 更省。
        self._index: list[dict[tuple[int, ...], list[int]]] = [
            {} for _ in range(max_match + 1)
        ]
        self._index_size = 0

    @property
    def index_size(self) -> int:
        """报告索引里有多少条 (窗口, 位置) 记录。

        读计数器而不是遍历字典，探针因此可以每步观察索引增长而不必走一遍。由
        ``observe`` 喂出来的草稿器和由 ``reset`` 喂出来的取值相同时，就证明增量
        路径索引的是同一份历史——这正是计数器陈旧时会藏掉的不变量。

        Returns:
            所有窗口长度上被索引位置的总数。
        """
        return self._index_size

    def reset(self, tokens: list[int]) -> None:
        """用 ``tokens`` 替换历史，并从头重建索引。

        在一个请求的开头、以及命中前缀缓存之后使用，此时复用的前缀整块到达。
        重建走的是 ``observe`` 用的同一个循环，两条路径因此不可能对「哪些位置
        存在」产生分歧。

        Args:
            tokens: 按顺序排列的请求 token，prompt 在前。传空列表会清空草稿器。
        """
        self._history = list(tokens)
        self._index = [{} for _ in range(self.max_match + 1)]
        self._index_size = 0

        # 从 reset 后的状态开始索引，而不是先清空再加一份旧索引的拷贝：只有一条
        # 代码路径，就只有一个「已定位」的概念。
        self._index_positions(0)

    def observe(self, tokens: list[int]) -> None:
        """把 ``tokens`` 追加到历史，只索引新增的位置。

        这是逐步调用的入口：一次解码步产出一个 token，所以这里的工作是
        ``max_match`` 次切片加插入，而不是重扫 prompt。

        Args:
            tokens: 新产出的 token，按顺序。空输入是空操作。
        """
        if not tokens:
            return
        start = len(self._history)
        self._history.extend(tokens)
        self._index_positions(start)

    def draft(self, k: int) -> list[int]:
        """预测最多 ``k`` 个延续历史末尾的 token。

        选取规则，按顺序：

        1. 后缀长度 ``m`` 从 ``max_match`` 试到 ``min_match``，取第一个在更早处
           出现过的长度。一次出现是一个结束位置 ``p``，其窗口
           ``history[p-m+1 : p+1]`` 等于该后缀，且 ``p <= len(history) - 2``，
           这样后缀不会匹配到它自己；只有后面还有 token 的位置才算数。
        2. 在该长度的所有出现中，取后缀之后最常跟随的那个 token 作为首份草稿；
           平票时选最近一次出现离末尾更近的那个，因为近期文本比旧文本更能预测
           下一个 token。
        3. 从这个首 token 离末尾最近的那次出现继续展开，于是扩展是真正的续写，
           而不是把两次不同的出现拼在一起。
        4. 返回 ``history[p+1 : p+1+k]``，在历史末尾处截断。

        草稿是已观测历史的一个切片，绝不是合成出来的，所以它在构造上不可能
        给出模型自己输出之外的 token。

        Args:
            k: 最多预测多少个 token。零或负数返回空，调用方不用分支就能关掉
                草稿。

        Returns:
            ``1`` 到 ``k`` 个 token；历史里找不出至少 ``min_match`` 个 token 的
            更早后缀时返回 ``[]``。
        """
        history = self._history
        n = len(history)
        if k <= 0 or n < self.min_match:
            return []

        # p == n - 1 就是后缀自身；允许它会把后缀自己窗口里紧跟其后的 token
        # 草稿出来。
        last_usable = n - 2
        for m in range(min(self.max_match, n), self.min_match - 1, -1):
            positions = self._index[m].get(tuple(history[n - m :]))
            if not positions:
                continue

            counts: dict[int, int] = {}
            latest: dict[int, int] = {}
            for p in positions:
                if p > last_usable:
                    continue
                token = history[p + 1]
                counts[token] = counts.get(token, 0) + 1
                if p > latest.get(token, -1):
                    latest[token] = p
            if not counts:
                continue

            # 先比频次，再比新旧；两个 token 不可能共用一个最新位置，所以这个
            # 次序是全序的，草稿可复现。
            first = max(counts, key=lambda token: (counts[token], latest[token]))
            best = latest[first]
            # best <= n - 2，所以 history[best + 1] 一定存在，k >= 1 的草稿
            # 总有首 token。
            return history[best + 1 : best + 1 + k]

        return []

    def _index_positions(self, start: int) -> None:
        """为所有长度索引那些在 ``start`` 或之后结束的窗口。

        只有结束位置在 ``start`` 及之后的窗口才是新的：更早的结束位置在它的
        token 到达时就索引过了，再碰一次会让它在出现列表里重复，后续频率投票时
        权重翻倍。

        Args:
            start: 其窗口可能为新的第一个历史下标。
        """
        history = self._history
        n = len(history)
        for end in range(start, n):
            for m in range(1, self.max_match + 1):
                if end + 1 < m:
                    continue
                position_list = self._index[m].setdefault(
                    tuple(history[end - m + 1 : end + 1]), []
                )
                position_list.append(end)
                self._index_size += 1
