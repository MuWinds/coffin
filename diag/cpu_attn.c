/* @file cpu_attn.c
 * @brief CPU-side decode attention probe for the tiered-KV design.
 *
 * Module: coffin/diag. Standalone probe: not part of the extension build, not
 * imported by anything. It answers one question -- how fast can a host core read
 * cold KV straight out of RAM -- so it shares no code with ``kernels/``.
 *
 * Geometry = Qwen3.8-27B full-attention layer: 24 q heads / 4 kv heads (rep=6) /
 * head_dim=256, int8 or int4 (g128) KV with per-(head,token) scale. This is the
 * "cold KV lives in host RAM and the CPU reads it directly" path: no PCIe hop, one
 * pass over K+V per layer, online softmax, split-K over the context with an
 * OpenMP parallel section over (kv head, key slice).
 *
 * Constraints:
 *   * AVX2 + FMA + OpenMP are required; the ``-mavx2 -mfma -fopenmp`` line below
 *     is the build contract, and a narrower target silently changes the numbers.
 *   * Every buffer is cache-line aligned (``aligned_alloc(64, ...)``) so the
 *     measured bandwidth is not limited by unaligned loads.
 *   * Output is one JSON line per run: the probe is read by the benchmark script,
 *     so keep the field names and print exactly one object.
 *   * The data is synthetic (xorshift fill, fixed scale): this measures bandwidth
 *     and instruction throughput, never numerical agreement with the GPU kernel.
 *
 * Build (pod):
 *   gcc -O3 -mavx2 -mfma -fopenmp -o /tmp/cpu_attn cpu_attn.c -lm
 * Run:
 *   OMP_NUM_THREADS=7 /tmp/cpu_attn int8 131072 8
 *   OMP_NUM_THREADS=7 /tmp/cpu_attn int4 131072 8
 *   OMP_NUM_THREADS=7 /tmp/cpu_attn read8 131072 8
 * Prints one JSON line per run.
 */
#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Geometry of the reference layer. NB/G are the int4 quantisation group: 2 groups
 * of 128 elements share one scale, which is what the g128 layout means. */
#define NKV 4
#define NH 24
#define REP 6
#define HD 256
#define NB 2
#define G 128

/* @brief Horizontal sum of one __m256: the reduce both dot products end with. */
static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

/* @brief dot(q[HD] float, k[HD] int8): the score of one key row.
 *
 * The int8 bytes are widened to int32 and then to float in one lane-group, so the
 * per-token scale is applied by the caller after the horizontal reduce.
 */
static inline float dot_i8(const int8_t *k, const float *q) {
    __m256 acc = _mm256_setzero_ps();
    for (int d = 0; d < HD; d += 8) {
        __m128i b = _mm_loadl_epi64((const __m128i *)(k + d));
        __m256i i = _mm256_cvtepi8_epi32(b);
        __m256 f = _mm256_cvtepi32_ps(i);
        acc = _mm256_fmadd_ps(f, _mm256_loadu_ps(q + d), acc);
    }
    return hsum256(acc);
}

/* @brief dot(q[HD] float, packed int4 row: HD/2 bytes, NB groups of G elements).
 *
 * Each group carries its own scale, so the group partials are scaled before the
 * final sum -- that is what makes the int4 path numerically comparable to int8
 * rather than a pure bandwidth number.
 */
static inline float dot_i4(const uint8_t *k, const float *q, const float *s) {
    float acc = 0.f;
    for (int g = 0; g < NB; ++g) {
        const uint8_t *kg = k + g * (G / 2);
        const float *qg = q + g * G;
        __m256 accg = _mm256_setzero_ps();
        for (int j = 0; j < G / 8; ++j) {   /* 8 elems = 4 bytes */
            uint32_t w;
            memcpy(&w, kg + 4 * j, 4);
            int u[8];
            for (int t = 0; t < 8; ++t) u[t] = (int)((w >> (4 * t)) & 0xF) - 8;
            __m256 f = _mm256_cvtepi32_ps(_mm256_setr_epi32(u[0], u[1], u[2], u[3],
                                                             u[4], u[5], u[6], u[7]));
            accg = _mm256_fmadd_ps(f, _mm256_loadu_ps(qg + 8 * j), accg);
        }
        acc += hsum256(accg) * s[g];
    }
    return acc;
}

/* @brief acc[HD] += w * v[HD] int8: the V side of one key, weight pre-folded. */
static inline void saxpy_i8(float *acc, const int8_t *v, float w) {
    __m256 wv = _mm256_set1_ps(w);
    for (int d = 0; d < HD; d += 8) {
        __m128i b = _mm_loadl_epi64((const __m128i *)(v + d));
        __m256 f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b));
        __m256 a = _mm256_loadu_ps(acc + d);
        _mm256_storeu_ps(acc + d, _mm256_fmadd_ps(f, wv, a));
    }
}

/* @brief acc[HD] += w * v4 dequantised (NB groups of G).
 *
 * The softmax weight and the group scale are folded into one multiplier first, so
 * the inner loop is a single FMA per eight elements.
 */
static inline void saxpy_i4(float *acc, const uint8_t *v, const float *s, float w) {
    for (int g = 0; g < NB; ++g) {
        const uint8_t *vg = v + g * (G / 2);
        const float wg = w * s[g];
        for (int j = 0; j < G / 8; ++j) {
            uint32_t x;
            memcpy(&x, vg + 4 * j, 4);
            int u[8];
            for (int t = 0; t < 8; ++t) u[t] = (int)((x >> (4 * t)) & 0xF) - 8;
            __m256 f = _mm256_cvtepi32_ps(_mm256_setr_epi32(u[0], u[1], u[2], u[3],
                                                             u[4], u[5], u[6], u[7]));
            float *p = acc + g * G + 8 * j;
            _mm256_storeu_ps(p, _mm256_fmadd_ps(f, _mm256_set1_ps(wg), _mm256_loadu_ps(p)));
        }
    }
}

/* One (kv head, key-slice) partial: the same online-softmax triple the GPU split
 * pass publishes, so the merge below mirrors ``attn_combine_kernel``. At ~6 KB per
 * entry the parallel tasks never share a cache line. */
typedef struct {
    float m[REP];
    float l[REP];
    float acc[REP][HD];
} Part;

/* @brief Monotonic clock in milliseconds, for the wall-time of one probe run. */
static inline double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

/* @brief Run one probe and print its JSON line.
 *
 * ``argv[1]`` selects the path: "int8" scores int8 rows, "int4" scores g128 int4
 * rows, "read8" skips attention and just streams K to measure the read ceiling the
 * other two must chase. ``argv[2]`` is the context length S and ``argv[3]`` the
 * split count (default 8), so one binary reproduces every point in HW_PROBE.md.
 *
 * @param argc At least 3; otherwise the usage line is printed and the process exits
 *             2 before touching any buffer.
 * @param argv Mode, context length, optional split count.
 * @return 0 on success, 1 when an allocation fails (the run is then meaningless
 *         rather than partial), 2 on bad usage.
 */
int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s int8|int4|read8 <S> [splits]\n", argv[0]);
        return 2;
    }
    const char *mode = argv[1];
    const long S = atol(argv[2]);
    const int split = argc > 3 ? atoi(argv[3]) : 8;
    const int is4 = strcmp(mode, "int4") == 0;
    const int read_only = strcmp(mode, "read8") == 0;

    const size_t row = HD;
    const size_t kbytes = (size_t)NKV * S * (is4 ? row / 2 : row);
    const size_t vbytes = kbytes;
    uint8_t *K = aligned_alloc(64, kbytes);
    uint8_t *V = aligned_alloc(64, vbytes);
    float *KS = malloc(sizeof(float) * NKV * S * (is4 ? NB : 1));
    float *VS = malloc(sizeof(float) * NKV * S * (is4 ? NB : 1));
    if (!K || !V || !KS || !VS) { fprintf(stderr, "alloc failed\n"); return 1; }
    uint32_t rng = 12345u;
    for (size_t i = 0; i < kbytes; ++i) {
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        K[i] = (uint8_t)(rng & 0xff);
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        V[i] = (uint8_t)(rng & 0xff);
    }
    for (size_t i = 0; i < (size_t)NKV * S * (is4 ? NB : 1); ++i) {
        KS[i] = 0.01f; VS[i] = 0.01f;
    }
    float q[NH][HD];
    for (int h = 0; h < NH; ++h)
        for (int d = 0; d < HD; ++d) q[h][d] = 0.05f * (float)((h * 131 + d * 7) % 17) - 0.4f;

    if (read_only) {
        volatile uint64_t s = 0;
        double t0 = now_ms();
        for (size_t i = 0; i < kbytes; i += 32) {
            __m256i a = _mm256_loadu_si256((const __m256i *)(K + i));
            s += (uint64_t)_mm256_extract_epi64(a, 0);
        }
        double ms = now_ms() - t0;
        printf("{\"mode\":\"read8\",\"S\":%ld,\"gbps\":%.1f,\"ms\":%.2f,\"sink\":%llu}\n",
               S, (kbytes / 1e9) / (ms / 1e3), ms, (unsigned long long)s);
        return 0;
    }

    const int parts = NKV * split;
    Part *p = calloc(parts, sizeof(Part));
    if (!p) { fprintf(stderr, "part alloc failed\n"); return 1; }
    const float scale = 1.0f / sqrtf((float)HD);

    double t0 = now_ms();
#pragma omp parallel num_threads(omp_get_max_threads())
    {
#pragma omp for schedule(static)
        for (int task = 0; task < parts; ++task) {
            const int g = task / split;
            const int si = task % split;
            const long k0 = (long)si * S / split;
            const long k1 = (long)(si + 1) * S / split;
            Part *pp = &p[task];
            for (int r = 0; r < REP; ++r) { pp->m[r] = -1e30f; pp->l[r] = 0.f; }
            memset(pp->acc, 0, sizeof(pp->acc));
            for (long s = k0; s < k1; ++s) {
                const uint8_t *kr = K + ((size_t)g * S + s) * (is4 ? row / 2 : row);
                const uint8_t *vr = V + ((size_t)g * S + s) * (is4 ? row / 2 : row);
                const float *ks = KS + (size_t)g * S * (is4 ? NB : 1) + (size_t)s * (is4 ? NB : 1);
                const float *vs = VS + (size_t)g * S * (is4 ? NB : 1) + (size_t)s * (is4 ? NB : 1);
                float sc[REP];
                for (int r = 0; r < REP; ++r) {
                    const int h = g * REP + r;
                    float dot = is4 ? dot_i4(kr, q[h], ks) : dot_i8((const int8_t *)kr, q[h]);
                    sc[r] = is4 ? dot * scale : dot * ks[0] * scale;
                }
                float mnew = pp->m[0];
                for (int r = 1; r < REP; ++r) mnew = fmaxf(mnew, sc[r]);
                float alpha = 0.f, sum = 0.f;
                for (int r = 0; r < REP; ++r) {
                    const float mm = fmaxf(pp->m[r], mnew);
                    alpha = expf(pp->m[r] - mm);
                    const float e = expf(sc[r] - mm);
                    sum += e;
                    pp->l[r] = pp->l[r] * alpha + e;
                    pp->m[r] = mm;
                }
                /* V accumulation: unaugmented numerator; scale applied per element */
                for (int r = 0; r < REP; ++r) {
                    const float w = expf(sc[r] - pp->m[r]);
                    if (is4) saxpy_i4(pp->acc[r], vr, vs, w);
                    else saxpy_i8(pp->acc[r], (const int8_t *)vr, w * vs[0]);
                }
            }
        }
    }
    double ms = now_ms() - t0;

    /* merge the slices (serial; tiny): the same rescale-then-sum the GPU combine
     * kernel does, over `split` partials instead of key blocks. */
    float out[NH][HD];
    for (int g = 0; g < NKV; ++g) {
        for (int r = 0; r < REP; ++r) {
            float m = -1e30f;
            for (int si = 0; si < split; ++si) m = fmaxf(m, p[g * split + si].m[r]);
            float l = 0.f;
            float a[HD];
            memset(a, 0, sizeof(a));
            for (int si = 0; si < split; ++si) {
                Part *pp = &p[g * split + si];
                const float alpha = expf(pp->m[r] - m);
                l += pp->l[r] * alpha;
                for (int d = 0; d < HD; ++d) a[d] += pp->acc[r][d] * alpha;
            }
            for (int d = 0; d < HD; ++d) out[g * REP + r][d] = a[d] / l;
        }
    }
    /* ``sink`` is printed so the compiler cannot elide the whole merge; it is not
     * a meaningful number. */
    double sink = 0;
    for (int h = 0; h < NH; ++h) sink += out[h][0];
    /* bytes counts the K+V rows plus both scale arrays, so "gbps" is the full read
     * traffic of the layer, not only the activations. "gflops" assumes two flops
     * per q.k and per weight.v element; "layers16_ms" scales one layer to the 16
     * full-attention layers of the reference model and "tok_s" is the implied
     * single-request token rate. */
    const double bytes = 2.0 * (double)NKV * S * (is4 ? HD / 2 : HD) + 2.0 * NKV * S * (is4 ? NB : 1) * 4;
    printf("{\"mode\":\"%s\",\"S\":%ld,\"split\":%d,\"threads\":%d,\"ms\":%.2f,"
           "\"gbps\":%.1f,\"gflops\":%.1f,\"layers16_ms\":%.0f,\"tok_s\":%.2f,\"sink\":%.6f}\n",
           mode, S, split, omp_get_max_threads(), ms, bytes / 1e9 / (ms / 1e3),
           (2.0 * NH * S * HD * 2.0) / 1e9 / (ms / 1e3), ms * 16, 1000.0 / (ms * 16),
           sink);
    return 0;
}
