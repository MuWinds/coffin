"""测试套件的导入跳板：MKL 线程层与 ``sys.path``。

Module: ``coffin/tests``。每个测试模块都最先导入它，所以它不能反过来导入其中
任何一个；它也是唯一改 ``sys.path`` 的测试文件。

Constraints:
    * ``MKL_THREADING_LAYER=GNU`` 必须在 MKL 初始化之前导出：容器只带了 GNU
      线程层，默认分派会以 FATAL ERROR 直接中止进程。torch 由测试模块自己 import，
      所以这条设置放在本文件最前面就够。
    * 各测试模块按扁平名字导入 ``support``，因此都必须走 ``discover -s tests``
      这套布局，让 ``tests/`` 进入 ``sys.path``；``coffin/`` 再由本文件加进去，
      这样 ``cache`` / ``engine`` / ``engine_api`` 才能按顶层包导入。
    * 这里只留这两件事。需要 checkpoint、具体模型或加速卡的用例按 ``AGENTS.md``
      第 10 条写在临时目录，不回到本文件里。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

#: ``coffin/`` 是所有包的 ``sys.path`` 根，``ops`` 与 ``models`` 才能是顶层包。
ENGINE_DIR = Path(__file__).resolve().parent.parent
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))
