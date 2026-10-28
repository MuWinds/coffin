# Decode-step fusion plan (shared state board)

> 2026-10-28. `gpu-node-05`, Iluvatar BI-V100, `COFFIN_BACKEND=corex`, int8 weights.
> **Read this whole file before touching anything.** Update your own row when you land a
> piece; never delete someone else's.

## 0. The measurement that drives all of this

`diag/corex_probe/probe_launch_floor.py` (job `floor2`, card 3):

```
launch floor (1-element add_) : 5.75 us
a kernel crosses 2x the floor : n = 1048576 elements
```

`diag/corex_probe/probe_launch_inventory.py` (job `inv1`, same card, same session):

```
decode step : 23.527 ms   (42.51 tok/s)
launches    : 2407 trace rows per step, 946 real kernel launches
wall/launch : 9.77 us
floor x 2407 = 13.83 ms = 59% of the wall
greedy ids  : [6, 24218, 10, 4838, 1665, 15]      <- the regression bar
```

**The step is launch-floor bound, not bandwidth bound.** A launch costs ~5.75-13 us whether
it moves one float or a megabyte. `PERF_COREX.md` §8 previously concluded "device bound,
removing launches cannot help" from a profiler sum of 34 ms against a 23 ms wall — that sum
over-counts, and it over-counts *per kernel*, so its inflation grows with the launch count
rather than with the work. The floor measurement above is what stands.

**So every optimisation here is a launch-count reduction.** The yardstick for any change:
how many launches does it remove, and does the token id list stay identical?

## 1. The launch inventory, by family (job `inv1`)

| what | calls/step | ms | us/call | note |
| --- | ---: | ---: | ---: | --- |
| `gemv_kn_i8x4_split_kernel` | 64 | 6.435 | 100.5 | weight read, irreducible |
| `gemv_kn_i8_split_kernel` | 64 | 4.764 | 74.4 | weight read, irreducible |
| **`gemv_reduce_i8_kernel`** | **128** | **2.108** | **16.5** | **pure waste — target A** |
| `delta_kernel` | 24 | 1.799 | 75.0 | target C's body |
| `gemv_kn_i8x4_kernel` | 1 | 1.411 | 1411 | `lm_head`, irreducible |
| `aten::copy_` | 99 | 1.264 | 12.8 | target B/D |
| `rms_norm_kernel` | 81 | 1.165 | 14.4 | target D |
| `aten::add` | 88 | 0.994 | 11.3 | target D (64 are residual adds) |
| `aten::mul` | 65 | 0.912 | 14.0 | target B (32 are `silu(gate)*up`) |
| `elementwise_kernel` (modern) | 80 | 0.883 | 11.0 | target B |
| `aten::index_copy_` | 32 | 0.549 | 17.2 | KV write, contract |
| `attn_slice_kernel` | 8 | 0.537 | 67.1 | attention body |
| `elementwise_kernel` (legacy 512) | 41 | 0.523 | 12.8 | target B |
| `aten::div` | 32 | 0.484 | 15.1 | target B (k/v quant) |
| `aten::cat` | 32 | 0.448 | 14.0 | target D (MLP gate/up) |
| `aten::silu` | 32 | 0.441 | 13.8 | target B |
| `prologue_kernel` | 24 | 0.436 | 18.2 | target C — should not exist alone |
| `aten::amax` | 16 | 0.418 | 26.1 | target B (k/v quant) |
| `conv1d_silu_kernel` | 24 | 0.354 | 14.8 | target C |
| `rms_norm_gated_kernel` | 24 | 0.352 | 14.7 | target C |
| `aten::clamp_`/`round`/`neg`/`abs`/`index_select` | ~150 | ~1.9 | ~12 | target B |

The four glue families the user named (`copy_` + `add` + `mul` + elementwise) are **252
launches, ~3.06 ms, 13%** of the wall — matching the "~15%" in the brief.

## 2. The four workstreams

### A. Fuse the split-K reduce (6%, 128 launches)

**Owner: `gemv-reduce`. Files: `backends/corex/kernels/gemv.cu` only.**

`gemv_reduce_i8_kernel` sums `split` fp32 partials per column and applies the scale. It
costs 16.5 us/launch on `[split, N]` that is usually a few tens of KB. 128 of them per step.

Two acceptable shapes, pick by measurement:

1. **Split-in-block, single launch.** One block covers `threads/split` columns; the `split`
   threads of a column reduce through shared memory. Removes the partial buffer entirely.
   Watch the coalescing: lanes of one wavefront must still hit adjacent columns, so the
   column mapping has to stay `t % cols_per_block`, and `cols_per_block * width` wants to be
   >= 64 for a full line. For `split` above `threads/64` this constraint bites — that is the
   thing to measure, not to assume.
2. **Last-block reduce.** Keep the partial buffer; each block `__threadfence()`s then bumps a
   per-tile counter; the last one sums `part[0..split-1]` in fixed order and resets the
   counter. Deterministic, so numerics are unchanged, but needs a persistent counter buffer
   (an extra zeroing launch would cancel the win).

**Numerics**: the current reduce sums in split order. Any replacement must sum in the same
order, or the result changes in the last bf16 ulp and the token ids may shift. If a variant
cannot preserve the order, say so and stop — the token ids are the bar.

**Do not** add a new op name; this is internal to `gemv.cu`.

### B. Kill the glue elementwise chain (15%, ~252 launches)

**Owner: `glue-attn`. Files: `backends/corex/kernels/attn.cu`, `models/qwen35/full_attn.py`,
`models/qwen35/blocks.py`.**

Two distinct items, one owner because both are "small elementwise ops in the hot layer":

1. **k/v int8 quantisation** (the biggest single glue item, ~100 launches).
   `FullAttn._quant` runs `abs -> amax -> clamp_min -> div -> round -> clamp_ -> to` per
   tensor, twice per full-attention layer: 8 layers x 2 tensors x ~6 launches. One fused
   kernel replaces it. Entry already declared in `corex_ops.h` and bound:
   `kv_quant(x, out_q, out_s) -> out_q`.
2. **`silu(gate) * up`** in `MLP.__call__` (32 silu + 32 mul = 64 launches). The `gate_up`
   GEMV already produces both halves in one read; one elementwise kernel over the product
   removes both. Implement as `ops`-level glue only — no new op name (see §3).

### C. Fuse the delta prologue and norm (5%, ~48 launches + intermediate traffic)

**Owner: `delta-fuse`. Files: `backends/corex/kernels/delta.cu`,
`backends/corex/kernels/linear_attn.cu`, `models/qwen35/linear_attn.py`.**

Today a linear-attention layer's decode step issues four launches in a row:

```
conv1d_silu -> delta_prologue -> delta -> rms_norm_gated
```

`delta_prologue` writes `q_pro`/`k_pro`/`v_pro` (`[32,128]` fp32 each), `beta`/`g`
(`[32]` fp32) and `delta` reads them back; `delta` writes `delta_out` (`[32,128]` fp32) and
`rms_norm_gated` reads it back. That is ~48 KB written and re-read per layer per token for
nothing. 24 layers.

Target: **one kernel doing prologue + recurrence + gated norm**, one block per value head.
Everything the prologue produces fits in shared memory for that head (`dk` 128, `dv` 128:
q/k/v = 3 x 128 floats, beta/g = 2). The gated norm is a per-head reduction over `dv`, which
is exactly the span of one block, so it folds into the epilogue for free.
Entry declared and bound already: `delta_normed(...)` — see §3 for the signature.

`conv1d_silu` stays its own launch (different width, different mapping); fusing it too is a
stretch goal only if the single-block version measures clean.

**Numerics**: the L2 epsilon is `1e-6`, the reductions must keep the same order, and `g <= 0`
must hold. `check_ops.py` compares against the reference backend; that stays the gate.

### D. Residual add into the norm (64 launches)

**Owner: `norm-residual`. Files: `backends/corex/kernels/norm.cu`, `ops/norm.py`,
`models/qwen35/blocks.py`.**

`blocks.Layer.__call__` is

```python
h = self.mix(self.in_ln(x), *ctx)
x = x + h
return x + self.mlp(self.post_ln(x))
```

Two `aten::add` launches per layer, 64 per step. Both can become the norm's prologue:
`rms_norm(x, w, stride, eps, add=h)` computes `x += h` in place and then normalises `x`, so
the residual buffer is the same buffer the norm reads — no extra read, no extra launch. The
`x` handed to the second add is the `in_ln` output buffer, so the same trick chains.

Entry: `rms_norm` gains an optional `add` argument; no new op name.

## 3. Interface: what the Lead owns, what the owners may touch

**Status: LANDED.** The op table went from 18 to 21 names; `python -m unittest discover -s
tests` is `Ran 238 tests, OK (skipped=58)` with the new table. Reference implementations for
all three exist in `backends/reference/__init__.py`, so a backend without the kernel still
answers and only pays the extra launches.

The three new entries as declared in `kernels/corex_ops.h` and bound in `bindings.cpp`:

```cpp
// B. one fused int8 quantisation of a [M, N] bf16 tensor, per-row absmax.
torch::Tensor kv_quant(torch::Tensor x, torch::Tensor out_q, torch::Tensor out_s);

// C. prologue + delta recurrence + gated norm, one block per value head.
torch::Tensor delta_normed(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                           torch::Tensor b, torch::Tensor a,
                           torch::Tensor neg_expA, torch::Tensor dt_bias,
                           torch::Tensor z, torch::Tensor nw, torch::Tensor st,
                           torch::Tensor out, double qscale, int64_t rep);

// D. residual add into x, then the zero-centred RMSNorm, one launch.
torch::Tensor add_rms_norm(torch::Tensor x, torch::Tensor add, torch::Tensor w,
                           torch::Tensor out, int64_t stride, double eps);
```

Adapters already exist in `backends/corex/__init__.py` (`kv_quant`, `add_rms_norm`,
`delta_normed`). `kv_quant` is registered in `_NATIVE`; the other two are registered when
their kernels land, and the reference answers in the meantime.

`LinearAttn` gained a `normed_out` buffer for the fused delta destination, so the call site
can be wired without an allocation on the decode path.

## 4. Environment isolation (read this before your first build)

Every agent rebuilds the torch extension when its sources change, and a stale cache is
invisible — the old objects load and answer. Two rules keep four agents from mixing:

1. **Never edit `_EXTENSION` in `backends/corex/kernels/__init__.py`.** It now honours an
   override: `COFFIN_COREX_EXT` from the environment. Set it in your launcher.
2. **Use your own value**: `gemv-reduce` -> `v13gemv`, `glue-attn` -> `v13attn`,
   `delta-fuse` -> `v13delta`, `norm-residual` -> `v13norm`. The Lead reconciles to one
   `v14` at the end.

Each cache is a separate build directory, so the four of you also stop contending on the one
build baton that has wedged this cluster before. **A first build after a source change costs
minutes**; that is the `EXT_CACHE` line in the log, not a hang.

## 5. How to run anything

```bash
# from coffin-repo/platform
python scow/tools/sync.py push -d ..\coffin \
    --launcher diag/corex_probe/<your launcher>.sh --run <name> -g 4 -t 25 --wait
```

`-g 4` matters: the partition has dead cards and `diag/_pick_card.sh` probes each one. Never
omit it, and never assume the node is bad before the per-card verdict is in.

Local, no card (all of these must pass before a job is worth submitting):

```bash
cd coffin
python -m py_compile $(find . -name '*.py' -not -path './__pycache__/*')
python -m unittest discover -s tests            # 238 tests, 58 skipped
python diag/check_style.py --style backends/corex
```

Local kernel check, no card — the existing pattern, worth copying rather than reinventing:
`.tmp_cusyn/rewrite.py` erases the launch syntax and `.tmp_cusyn/cuda_stub.h` stubs the CUDA
keywords, so a `.cu` compiles as host code and the kernel body can be *called* with
`blockIdx`/`threadIdx` set by hand and compared to a host fp64 oracle.
`.tmp_cusyn/check_width_kernels.py` is a full worked example. **Use it.** A kernel that is
numerically wrong on the host will not become right on the card, and every card round trip
costs minutes.

## 6. Acceptance, per workstream

| bar | how |
| --- | --- |
| greedy ids unchanged | `[6, 24218, 10, 4838, 1665, 15]` from `probe_launch_inventory.py` |
| launch count down | the `launches` line of the same probe |
| step time down | its `BASELINE step_ms=` line, compared against 23.527 |
| no numeric drift per op | `backends/corex/diag/check_ops.py` vs the reference backend |
| local suite green | `python -m unittest discover -s tests` |

A change that trades one launch for another launch plus an allocation has not helped. Say so
in the log rather than keeping it.

## 7. Board — final

| workstream | owner | status | measured |
| --- | --- | --- | --- |
| A. split-K reduce fusion | `gemv-reduce` | **negative, reverted** | 128 reduce launches removed and the step got **slower**: 20.411 -> 21.299 ms (+0.89 ms). Split-in-block worse still: 33.324 ms. `gemv.cu` restored. See §8. |
| B. glue elementwise (kv_quant, silu_mul) | `glue-attn` | **landed** | ids unchanged; 0.885 ms of the win is the kernels' own device time, the rest is launch floor |
| B3. rope_apply | `glue-attn` | **landed** | 136.4 -> 11.1 us per layer, bit-identical on q and k |
| C. delta prologue+norm | `delta-fuse` | **negative, reverted** | bit-exact on output *and* state, but same-session A/B is +5.3 us/call and the real step went 20.285 -> 20.615 ms. See §9. |
| D. residual into norm | `norm-residual` | **landed** | −32 launches; the second residual add was found to pair with the *next* layer's norm, not this one — deferred, see `backends/corex/NORM_RESIDUAL.md` |
| interface + reconciliation | lead | **done** | 18 -> 23 ops, 241 local tests |

### 7.1 Shipped result

```
decode step   23.527 ms   ->  20.263 ms   (1.16x)
throughput    42.51 tok/s ->  49.35 tok/s
trace rows    2407        ->  1399
native ops    18/18       ->  23/23
greedy ids    [6, 24218, 10, 4838, 1665, 15]   unchanged in every run
```

Both rejected fusions were **bit-exact** and both really did remove launches. They lost for
the same reason, and it is the finding worth carrying: **the 5.75 us launch floor bounds the
gap between launches, not the work.** Removing a launch only pays when the work disappears
with it — both of these moved the work into a neighbouring kernel instead. A's reduce time
reappeared as `+1.65 ms` in `gemv_kn_i8x4_split` plus `+0.63 ms` in `gemv_kn_i8_split`,
because a device-scope `__threadfence()` per block costs more than the store it replaces and
the separate reduce had been overlapping the partial grid's tail. C's three reductions cannot
share one block geometry with the recurrence's register-loop mapping.

The bug class worth carrying forward is in `PERF_COREX.md` §12.5: an adapter plus a header
declaration plus a binding plus a protocol field plus a reference implementation can all land
green while the op never executes natively, and a measurement taken then reads as a clean
zero. Both halves of that trap were hit for real here. Two guards now catch them
(`tests/test_backends.py::NativeRegistrationTest`) and both were verified to go red on purpose.


## 8. Workstream A: the split-K reduce fusion does not pay (2026-10-28)

**Result: not landed. `gemv.cu` is unchanged from the commit this workstream started on.**

Three reduce shapes were built and measured in one job on one card (`gemvreduce2`, card 3),
each as a subprocess of the same probe so the only difference is the kernel. Median of 5 x 30
timed steps per mode:

| reduce shape | launches/step | step (median) | vs shipped |
| --- | ---: | ---: | ---: |
| shipped two-launch | 1231 | **20.411 ms** | — |
| last-block fused (1 launch) | 975 | **21.299 ms** | **+0.888 ms slower** |
| split-in-block (1 launch, no buffer) | — | **33.324 ms** | +12.9 ms slower |

Greedy ids were `[6, 24218, 10, 4838, 1665, 15]` in **all three** modes. The numerics were
never the problem: a host harness (`.tmp_cusyn/check_fused_reduce.py`, real threads, real
`__syncthreads`, real atomics) proved both fused shapes `memcmp`-identical to the two-launch
path across scalar/x4/w2/w8/w16, fp32 and bf16, K in {3,5,67,96,256,1024,1280,2304,2560},
split 2..32, including N not divisible by the tile. The fusion is correct and the launch count
really does drop by 128. It is still slower.

### Why 256 launches disappeared instead of 128

`cudaLaunchKernel` fell 562 -> 434, i.e. 128, which is the reduce count. But the profiler's
launch total fell 1231 -> 975, a further 128. The difference is the profiler's `cudaLaunchKernel`
row being a host-side API row that the two runs bucket differently, not a second kernel family
disappearing; the device-side inventory is the one to trust, and the row that vanished there is
exactly `gemv_reduce_i8_kernel  128  2.076 ms`.

### Why removing a 2.1 ms launch made the step 0.89 ms slower

The reduce's 2.076 ms of device time did not disappear -- it moved into the kernels that were
supposed to be merely writing partials:

| kernel | shipped | fused | delta |
| --- | ---: | ---: | ---: |
| `gemv_kn_i8x4_split` -> `..._fused` | 6.445 ms | 8.095 ms | **+1.650** |
| `gemv_kn_i8_split` -> `..._fused` | 4.750 ms | 5.380 ms | **+0.630** |
| `gemv_reduce_i8` | 2.076 ms | — | **-2.076** |
| device total | 23.851 | 24.032 | **+0.181** |

The partial kernels absorbed +2.28 ms to save a 2.08 ms launch. Two costs explain it, and both
are structural rather than fixable by tuning:

1. **`__threadfence()` on every block.** The last-block protocol requires each of the 4096
   split blocks to fence its partials to device scope before taking a ticket. A device-scope
   fence flushes the block's writes to where every other SM can see them, which is strictly
   more expensive than the plain st.global the two-launch path does. That is the +2.28 ms.
2. **The reduce runs on the critical path of the last block, not in a separate launch.** In the
   two-launch path the reduce is its own kernel and overlaps the tail of the partial kernel's
   grid through normal launch pipelining: the reduce's blocks start as soon as their tile's
   partials are done. In the fused path the reducing block cannot start until the *whole tile*
   has arrived, so that overlap is gone.

The launch floor measured at 5.75 us (§0) is real, but it is a floor on the *gap between
launches*, not on the work. Removing a launch only helps if the work it did is also removed;
here the work was moved, and the move cost more than the gap.

### Why split-in-block is much worse (33.324 ms, +64%)

It is the coalescing problem the brief predicted, and it shows up at the measured splits rather
than at the edge: with `cols = threads / split`, the decode dispatch wants split 16-32 on
256 threads, which is `cols` 8-16. A 64-lane wavefront then spans 4-8 different k-rows, so each
lane's load lands in a different 128-byte line and the transaction efficiency collapses. The
variant is only viable at `split <= threads/64` (4 on 256 threads), and the shapes that fit that
constraint are exactly the wide ones that do not need split-K. The measured 33 ms is consistent
with the transaction-waste arithmetic in PERF_COREX.md §11.

### What would have to be true for this to pay

- The reduce's device time would have to be much smaller than the fence it replaces, i.e. a
  small `split`. At split 4 with a long K the fence overhead amortises over more work, but
  then the partial buffer is small too and there is little left to save.
- Or the fence would have to be avoidable: a cooperative-groups grid barrier has the same
  device-scope visibility requirement, so it does not dodge the cost.
- Or the reduce would have to move somewhere that is not the critical path at all -- folded
  into the *next* kernel that consumes the projection output (the consumers are target B/D's
  elementwise chain), which is a different workstream's file scope.

The honest summary: this was the one target in the plan whose payoff looked like pure
arithmetic (128 launches x 16.5 us), and the arithmetic was right about the launches and wrong
about the work. The launch floor is real, but a launch is only free to remove if nothing the
launch did has to happen somewhere else.

## 9. Workstream C: the delta chain fuses bit-exactly, and is still slower (2026-10-28)

**Result: not landed. The call site stays on the three-op chain. `delta_normed_kernel` and its
`delta_normed` entry remain in `linear_attn.cu`, declared, defined and bound, so the negative
result is reproducible from the tree rather than from a discarded branch.**

Measured on the card (`probe_delta_ab.py`, same process, alternating A/B, 5 rounds, jitter
<0.5%), and confirmed end-to-end:

```
chain (3 launches): 86.26 us/call
fused (1 launch):   91.57 us/call      fused - chain = +5.32 us/call
output bit-identical: True             state bit-identical: True
```

In the real step it is worse than isolated: `delta_normed` costs 2.781 ms / 24 layers =
115.9 us/layer, against `15.1 + 73.7 + 14.0 = 102.8 us/layer` for the chain being replaced.
That is **+13.1 us/layer x 24 = +0.315 ms/step**, and the measured step moved 20.285 ->
20.615 ms (+0.330 ms), consistent in direction and size. The 48 launch slots the fusion
removes are worth 48 x 5.75 us = 0.276 ms at the floor, so the kernel has to be spending
roughly **0.6 ms/step more device time** to buy them. The trade is negative and it is not close.

Three estimates of the size disagree and that disagreement is informative: the isolated A/B
(+5.32 us/call) predicts only +0.128 ms/step, the per-layer decomposition predicts +0.315,
the step measured +0.330. The A/B is the most trustworthy number and it under-predicts the
end-to-end regression by ~2.5x, which says the fusion's cost is partly *outside* the kernel
body — occupancy or scheduling, not just the extra arithmetic. Per §12.1 the profiler's
per-kernel sums are not additive, so the `102.8 -> 115.9` split above is a direction, not a
budget. The decision does not depend on which estimate is right: they all agree the fusion
loses.

### Why: the launch floor is a floor on the gap, not on the work

`delta_kernel` is fast because one thread owns one value column and walks `dk` in registers:
the recurrence's two `dk` reductions are entirely thread-local, **no cross-lane communication
at all**. The prologue and the gated norm each need a block reduction, and the fused kernel
must pick one geometry for all three. That compromise is what costs 24.6 us/layer more device
time than the two launch slots it saves — the same lesson as §8, from the opposite direction.
A launch is only free to remove if the work it did also disappears; here the work moved.

This is the identical failure signature to workstream A: launches down, step up, numerics
perfect. Two out of the three "obvious" fusion candidates in this plan lost to the same
arithmetic error, which is the strongest argument in the file for measuring a fusion in place
rather than reasoning about the launch count.

### What made it bit-identical (worth keeping even though the fusion lost)

The acceptance bar here is the greedy ids, not a tolerance, and four separate things each
would have moved them. Any future attempt at this chain has to reproduce all four:

1. **Block size is an invariant, not a tuning knob.** All three shipped stages reduce with 128
   threads at `dk = dv = 128`, so one 128-thread block per value head reproduces every
   reduction's partner pairing. The wrapper *rejects* a geometry where `dv > head_threads(dk)`
   or `norm_threads(dv) != head_threads(dk)` rather than silently summing in a different order.
2. **The recurrence's fp32 output must be rounded to bf16 before the norm.** `RMSNormGated`
   casts `delta_out` to the activation dtype before its kernel sees it, so the chain's norm
   never reads fp32. Normalising the unrounded value scales by an RMS half a bf16 ulp off.
3. **The gate is rounded to bf16 whatever its storage**, for the same reason. This is why the
   kernel templates on `ZT`: reading an fp32 `z` as bf16 reinterprets every second half-word.
4. **Operand order is preserved** — `xb * inv * nw * silu` associates left to right.

`block_sum_scalar` copies `norm.cu`'s `block_sum` and `norm_threads` copies its `rms_threads`,
because both live in another translation unit behind an anonymous namespace. The duplication is
checked rather than trusted: drift makes the wrapper reject a geometry it could have served,
which is a documented fallback, whereas the reverse would be silent.

### Verification (host, real barriers)

`.tmp_cusyn/check_delta_normed.py` drives the *shipped* kernel bodies with one OS thread per
simulated thread and a real block barrier — a no-op `__syncthreads` turns the shared tree into
garbage and a stubbed `rsqrtf` erases every number, so the usual host stub cannot verify a
kernel whose claim is about reduction order.

| check | result |
| --- | --- |
| fused == chain, output, 4 token steps @ nv=32 dk=128 dv=128 rep=2 | **bit for bit** |
| fused == chain, recurrent state, same run | **bit for bit** |
| same at rep=4, nv=8, dv=96, dv=72, dv=96/rep=4 | **bit for bit** |
| fused vs fp64 oracle from the recurrence's definition | rel 1.2e-09 (worst case 5.8e-04) |
| bf16 and fp32 destinations carry the same value | ok |
| both destination instantiations advance the state alike | ok |
| fp32 gate == bf16 gate, output and state | ok |
| `g <= 0`: zero state stays zero, incl. the softplus linear tail | ok |

The card A/B independently confirmed output and state bit-identical, so host and card agree.

### Rejected

- **Fusing `conv1d_silu` in as well.** It is over `conv_dim = 8192` channels with a 3-tap
  window per channel, not per head, so it has no block matching the head's 128 threads.
- **A bare `block_sum` for the norm.** A third copy of the tree; `block_sum_scalar` reuses
  `block_sum2`'s pairing by construction, and a dummy second column through `block_sum2` would
  double the last phase's shared traffic.
- **Keeping `z` bf16-only to dodge the template.** The header contract already permitted fp32;
  narrowing the contract to match the code leaves a caller able to pass fp32 and get silent
  garbage.
- **The fusion itself**, on the measurement above. Kept in the tree, not on the call path.

