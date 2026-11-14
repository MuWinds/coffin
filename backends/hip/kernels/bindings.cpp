// @file bindings.cpp
// @brief 扩展的模块入口：把自注册表里的绑定逐个装到模块上，这里不再写任何一条绑定。
//
// Module: coffin/backends/hip/kernels。内核的名字、参数名与默认值都写在定义它的 ``.cu``
// 里（``COFFIN_BIND``），本文件只剩模块定义本身；这里出现的名字与 ``hip_ops.py`` 之间
// 仍是字符串契约，只是契约的正文不再集中在这一处。
//
// 为什么模块入口还得单独存在：``PYBIND11_MODULE`` 决定扩展的模块名（来自
// ``TORCH_EXTENSION_NAME``，也就是 ``shim.py`` 加载时给的名字），并且是唯一能遍历注册表
// 的地方。它留在纯主机编译单元里，不放进任何一个 ``.cu``：模块初始化与设备编译器无关。
//
// Constraints:
//   * 一个名字只能注册一次。撞名时 pybind 会在初始化期抛 ``already registered``，而那要
//     等到导入扩展才看得见——所以不要再往这里加 ``m.def``，多一条就回到两处维护。
//   * ``hip_ops.h`` 的声明仍是 ``.cu`` 的编译前提，但这里不再直接绑定它；不要再 include 它
//     来「顺手核对名字」。

#include "hip_bindings.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    for (const auto binder : coffin_hip::binders()) {
        binder(m);
    }
}
