"""引擎可以服务的模型实现。

每个架构族一个子包。一个族提供：

* 一个实现 ``engine_api.Model`` 契约的模型类（当它的层带循环状态时，还要
  实现 ``engine_api.PagedModel`` 的分页 / 混合能力），以及
* 一个 ``load_weights`` 函数，签名与服务端使用的一致，以便重载时张量保持常驻。

``engine_api`` 负责从 checkpoint 配置到子包的映射。

Module: ``coffin/models``。模型侧的最底层；不从 ``engine_api`` 或服务层导入
任何东西，所以新增一个族不必碰它们。

Constraints:
    * 导入期不做任何工作：import 这个包不许加载权重、分配设备内存，也不许
      导入需要 DCU 的 ``torch`` 子模块。
    * 族子包由 ``engine_api`` 的注册表发现，不是靠这里的副作用，所以新增一个
      永远不需要改这个文件。
"""
