"""常驻引擎和 HTTP 适配器的命令行入口。

Module: ``coffin.entrypoints.cli``。``runtime`` 拉起模型进程，``openai``
只起 HTTP 层并连它的 socket。

Constraints:
    * 两个入口是分开的进程：HTTP 侧重启不该连带重载模型。
"""
