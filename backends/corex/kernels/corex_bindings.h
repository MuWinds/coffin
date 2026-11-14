// @file corex_bindings.h
// @brief 内核自注册表：每条 pybind 绑定写在内核自己的翻译单元里，``bindings.cpp`` 只遍历。
//
// Module: coffin/backends/corex/kernels。被本目录的 ``.cu`` 与 ``bindings.cpp`` 包含；
// 它经由 ``corex_ops.h`` 带进 ``torch/extension.h``（即 pybind11），所以包含它的翻译单元
// 必须能编主机代码。这里的编译器是主机的 clang，不是设备编译器。
//
// 为什么不是一张写在 ``bindings.cpp`` 里的名字表：那个文件与内核定义分在两个翻译单元，
// 中间还隔着 ``corex_ops.h`` 的声明，加一个算子要按同样的签名抄三遍（声明、``m.def``、
// 适配器），而抄错的那一处只在卡上表现为 ``AttributeError``——``delta_normed`` 就是这么
// 丢的。把绑定放到内核旁边之后，名字、参数名与默认值只有一份，位置也只差几十行。
//
// Constraints:
//   * 注册发生在静态初始化期，表里存的是函数指针，不是 pybind 对象。
//     ``pybind11::arg("x") = 0`` 这类默认值必须等到解释器存在、模块正在初始化的那一刻
//     才构造；静态期碰 ``py::object`` 会把引用计数留到解释器退出时才回收，那里已经没有
//     解释器了。
//   * ``##__VA_ARGS__`` 是 GNU 扩展（clang 与 gcc 都支持）：没有额外参数时它吃掉前面那个
//     逗号，所以 ``COFFIN_BIND(quant_act_i8, "...")`` 展开成合法的三参数 ``m.def``。换成
//     ``__VA_OPT__`` 要求 C++20，而扩展按 C++17 编译。
//   * 表是函数内静态变量，取用它的函数是 ``inline``：跨翻译单元只有一份，也不依赖静态
//     初始化顺序——第一个登记它的翻译单元会把表建出来。
//   * 宏只做「同名绑定」：绑定名固定是函数名。这不是限制，只是约定——内核以别的名字
//     露给 Python 时用 ``COFFIN_BIND_LAMBDA(name, &fn, ...)``，别为了迁就宏去造一个名字
//     对不上实现的转发函数。

#pragma once

#include "corex_ops.h"

#include <vector>

namespace coffin_corex {

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

}  // namespace coffin_corex

// @brief 把 ``fn`` 绑定成 Python 名字 ``#fn``，并登记进自注册表。
//
// 用法与 ``m.def`` 逐字相同，只是少写一遍名字、也不再需要 ``bindings.cpp`` 里那一行：
//
//     COFFIN_BIND(rms_norm, "fused zero-centred RMSNorm over pitched [M, N] rows",
//                 pybind11::arg("x"), pybind11::arg("w"),
//                 pybind11::arg("stride") = 0, pybind11::arg("eps") = 1e-6);
//
// @param fn 内核的主机侧入口，必须是与 Python 侧同名的全局函数；名字就是绑定名。
// @param doc 文档字符串，进程内唯一的 API 文档。
// @param ... ``pybind11::arg(...)`` 形式的参数名与默认值，可省略。
#define COFFIN_BIND(fn, doc, ...)                                \
    static void coffin_bound_##fn(pybind11::module_& m) {         \
        m.def(#fn, &fn, doc, ##__VA_ARGS__);                     \
    }                                                            \
    [[maybe_unused]] static const bool coffin_registered_##fn =  \
        ::coffin_corex::add(&coffin_bound_##fn);

// @brief 同 `COFFIN_BIND`，但绑定体是一个 lambda 或任何可调用对象，不要求它叫这个名字。
//
// 只有「绑定必须做一次形状转换」的入口用得上它：内核的主机侧入口拿的是 ``char*`` 缓冲，
// 而 Python 面要的是 ``(int, str)``，中间那层 lambda 没有名字，也不该为了能被宏接住而
// 硬造一个名字。
//
// @param name 绑定名。
// @param callable 可调用体，可以直接是 lambda。
// @param doc 文档字符串。
// @param ... ``pybind11::arg(...)`` 形式的参数名与默认值，可省略。
#define COFFIN_BIND_LAMBDA(name, callable, doc, ...)                      \
    static void coffin_bound_lambda_##name(pybind11::module_& m) {        \
        m.def(#name, callable, doc, ##__VA_ARGS__);                      \
    }                                                                    \
    [[maybe_unused]] static const bool coffin_registered_lambda_##name =  \
        ::coffin_corex::add(&coffin_bound_lambda_##name);
