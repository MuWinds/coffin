"""常驻推理运行时：模型进程、它的 socket 和请求循环。

运行时实现都在这里；进程入口在 ``entrypoints/``。

Module: ``coffin.serving``。位于依赖图顶端：它驱动 ``engine``（请求、调度、
KV）、``backends``（设备和算子）和 ``utils``（分词器侧辅助），除了入口之外
没有模块 import 它。``runtime_transport`` 是唯一碰 socket 的模块，
``runtime_support`` 放的是没有加速卡也必须能跑的部分。

Constraints:
    * 模型由单个引擎线程独占：连接只提交任务并等待各自的队列，任何请求
      处理函数都不会自己跑前向。
    * 这里绝不点名具体平台。凡是与设备相关的部分——命名空间、显存读数、
      计时屏障、能否 capture——一律从 ``backends`` 取，正因为这样
      ``COFFIN_BACKEND=reference`` 才能在没卡的机器上启动。
"""
