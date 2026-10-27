// @file corex_scalar.h
// @brief Bit-level bf16 conversions shared by the CoreX device kernels.
//
// Module: coffin/backends/corex/kernels. Included by the ``.cu`` files only, never by
// ``bindings.cpp``: it is device code, and the binding translation unit is compiled by
// the host compiler, which has no ``__device__``.
//
// Constraints:
//   * bf16 is carried as a ``uint16_t`` bit container and converted by shifting, not
//     through ``cuda_bf16.h``. The conversion is then bit-identical to the gfx906
//     kernels' and needs no vector-types header, which is one less thing the CoreX
//     clang toolchain has to support.
//   * Narrowing rounds to nearest even, matching what a torch cast to bf16 does. A
//     truncating shift would bias every projection low, and the bias accumulates over
//     32 layers.
//   * fp32 is the accumulation dtype everywhere. This device's fp64 is "limited
//     support" and returns zeros (COREX_PORT.md §0), so it cannot be the accumulator
//     even where it would be more accurate.

#pragma once

#include <cstdint>

// @brief Conversions between one storage type and the fp32 a kernel accumulates in.
//
// Two specialisations cover every instantiation the kernels need: bf16 weights and
// activations, and fp32 for the fallback path a caller may hold tensors in.
template <typename T>
struct CxScalar;

template <>
struct CxScalar<uint16_t> {
    // @brief Widen one bf16 bit pattern to fp32.
    static __device__ __forceinline__ float to_float(uint16_t v) {
        return __uint_as_float(((unsigned)v) << 16);
    }

    // @brief Narrow fp32 to bf16, round to nearest even.
    static __device__ __forceinline__ uint16_t from_float(float v) {
        unsigned u = __float_as_uint(v);
        u += 0x7fffu + ((u >> 16) & 1u);
        return (uint16_t)(u >> 16);
    }
};

template <>
struct CxScalar<float> {
    // @brief Identity: the fp32 instantiation widens nothing.
    static __device__ __forceinline__ float to_float(float v) { return v; }

    // @brief Identity: the fp32 instantiation narrows nothing.
    static __device__ __forceinline__ float from_float(float v) { return v; }
};

// @brief int8 weight storage: the W8A16 GEMV reads one signed byte per weight.
//
// The pair to this specialisation matters: an int8 weight arrives with a per-column
// fp32 scale that the kernel applies once to the accumulated dot product, so the
// conversion here must stay the plain integer widening. Folding anything else into
// ``to_float`` would apply a scale twice or in the wrong place.
template <>
struct CxScalar<int8_t> {
    // @brief Widen one signed weight byte to fp32.
    static __device__ __forceinline__ float to_float(int8_t v) { return (float)v; }

    // @brief Narrow fp32 to int8 by truncation.
    //
    // Never used to store a weight -- quantisation rounds through
    // ``int8_weights.QWeight.quantize`` on the host -- but a specialisation must be
    // complete for the generic kernel template to instantiate.
    static __device__ __forceinline__ int8_t from_float(float v) { return (int8_t)v; }
};
