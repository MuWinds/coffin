"""离线运维工具：把 checkpoint 变成别的形状或格式，与引擎运行无关。

Module: ``coffin/tools``。导入图的旁支：这里的模块可以 import 引擎的叶子模块
（``int8_weights``），但引擎的任何代码都不 import 它，所以工具出问题不该能把
服务拖下水。

Constraints:
    * 只放能独立跑的脚本，不放库代码。被服务路径需要的东西属于 ``ops`` /
      ``models`` / ``backends``，放进来就会被 import，而工具是会随运维需要改的。
    * 每个脚本都要能被 ``python -m tools.<name>`` 直接调用：``coffin/`` 是
      ``sys.path`` 根，这个包因此与 ``ops``、``models`` 平级。
"""
