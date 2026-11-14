// @file bindings.cpp
// @brief 扩展的模块入口：把自注册表里的绑定逐个装到模块上，这里不再写任何一条绑定。
//
// Module: coffin/backends/corex/kernels。内核的名字、参数名与默认值都写在定义它的
// ``.cu`` 里（``COFFIN_BIND`` / ``COFFIN_BIND_LAMBDA``），本文件只剩模块定义本身。
// 这些名字与 ``backends/corex/kernels/__init__.py`` 之间仍是字符串契约，只是契约的正文
// 不再集中在这一处。
//
// 为什么模块入口还得单独存在：``PYBIND11_MODULE`` 决定扩展的模块名（来自
// ``TORCH_EXTENSION_NAME``），并且是唯一能遍历注册表的地方。它留在纯主机编译单元里，
// 不放进任何一个 ``.cu``：模块初始化与设备编译器无关，不该因为某张卡的编译期状况而
// 变化。
//
// Constraints:
//   * 一个名字只能注册一次。撞名时 pybind 会在初始化期抛 ``already registered``，而那
//     要等到导入扩展才看得见——所以不要再往这里加 ``m.def``，多一条就回到两处维护。
//   * ``py::arg`` 的默认值在内核那边的注册函数里求值，也就是模块正在初始化的时刻；
//     这里不参与，也不该替它补默认值。

#include "corex_bindings.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    for (const auto binder : coffin_corex::binders()) {
        binder(m);
    }
}
