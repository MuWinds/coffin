// @file hip_bindings.h
// @brief 内核自注册表：每条 pybind 绑定写在内核自己的翻译单元里，``bindings.cpp`` 只遍历。
//
// Module: coffin/backends/hip/kernels。被本目录的 ``.cu`` 与 ``bindings.cpp`` 包含；它经
// 由 ``hip_ops.h`` 带进 ``torch/extension.h``（即 pybind11），所以包含它的翻译单元必须能
// 编主机代码。
//
// 为什么不是一张写在 ``bindings.cpp`` 里的名字表：那个文件与内核定义分在两个翻译单元，
// 加一个算子要按同样的签名抄三遍（``hip_ops.h`` 的声明、``m.def``、内核实现），漏抄的那
// 一处只在加载扩展时表现为 ``AttributeError``。把绑定放到内核旁边之后，名字、参数名与
// 默认值只有一份，位置也只差几十行。
//
// Constraints:
//   * 注册发生在静态初始化期，表里存的是函数指针，不是 pybind 对象。``py::arg`` 的默认
//     值必须等到解释器存在、模块正在初始化的那一刻才构造；静态期碰 ``py::object`` 会把
//     引用计数留到解释器退出之后。
//   * ``##__VA_ARGS__`` 是 GNU 扩展（clang 与 gcc 都支持）：没有额外参数时它吃掉前面那个
//     逗号，所以 ``COFFIN_BIND(rms_norm, "...")`` 展开成合法的三参数 ``m.def``。
//   * 表是函数内静态变量，取用它的函数是 ``inline``：跨翻译单元只有一份，也不依赖静态
//     初始化顺序。
//   * 宏只做「同名绑定」：绑定名固定是函数名。这几个 HIP 入口全部满足该前提，所以没有
//     对应的 lambda 变体；将来真需要包一层时再补，不要先把用不上的东西写进契约。

#pragma once

#include "hip_ops.h"

#include <vector>

namespace coffin_hip {

// @brief 一次绑定：把一个原本要写在 ``PYBIND11_MODULE`` 里的注册动作变成可以搬运的值。
using Binder = void (*)(pybind11::module_&);

// @brief 注册表本体。函数内静态变量，第一次被用到时构造。
inline std::vector<Binder>& binders() {
    static std::vector<Binder> registry;
    return registry;
}

// @brief 登记一条绑定。
//
// @param binder 形如 ``void(pybind11::module_&)`` 的注册函数。
// @return 恒为 ``true``，只为了让静态初始化器能写成一个初始化表达式。
inline bool add(Binder binder) {
    binders().push_back(binder);
    return true;
}

}  // namespace coffin_hip

// @brief 把 ``fn`` 绑定成 Python 名字 ``#fn``，并登记进自注册表。
//
// 用法与 ``m.def`` 逐字相同，只是少写一遍名字、也不再需要 ``bindings.cpp`` 里那一行：
//
//     COFFIN_BIND(gemv_bf16, "bf16 GEMV (R rows per warp, U-way K-loop unroll)");
//
// @param fn 内核的主机侧入口，必须是与 Python 侧同名的全局函数；名字就是绑定名。
// @param doc 文档字符串，进程内唯一的 API 文档。
// @param ... ``pybind11::arg(...)`` 形式的参数名与默认值，可省略。
#define COFFIN_BIND(fn, doc, ...)                                \
    static void coffin_bound_##fn(pybind11::module_& m) {         \
        m.def(#fn, &fn, doc, ##__VA_ARGS__);                     \
    }                                                            \
    [[maybe_unused]] static const bool coffin_registered_##fn =  \
        ::coffin_hip::add(&coffin_bound_##fn);
