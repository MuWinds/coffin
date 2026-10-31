"""OpenAI HTTP 适配器的稳定入口。

真正的实现在 ``app`` 里；这里只做包级别的重定向，让
``python -m entrypoints.openai.server`` 这类调用路径不必随内部拆包而改动。

Module: ``coffin.entrypoints.openai``。

Constraints:
    * 最后一行把 ``sys.modules[__name__]`` 换成 ``app`` 模块，所以此后
      「本模块」就是 ``app``。任何新增定义都必须写在这次替换之前，否则
      外部看到的是旧模块对象。
"""

from . import app as _impl
from entrypoints.cli.openai import main
import sys as _sys

_impl.main = main
_sys.modules[__name__] = _impl

if __name__ == "__main__":
    _impl.main()
