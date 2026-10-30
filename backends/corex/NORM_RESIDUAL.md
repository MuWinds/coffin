# Folding the residual add into the RMSNorm

> 2026-10-28, `gpu-node-05`, Iluvatar BI-V100, `COFFIN_BACKEND=corex`.
> Workstream D of `../../../PERF_COREX.md`. **Partially landed: patch A is in, patch B is not.**

## What landed

`add_rms_norm(x, add, w, out, stride, eps)` in `kernels/norm.cu`, the
`RMSNorm.add_residual(x, add, out)` entry point in `ops/norm.py`, and one call site:
`blocks.Layer.__call__` (patch A). That removes the **first** of the layer's two
residual adds — 32 launches per decode step.

The **second** add is still a separate `aten::add`. It is *not* folded, and the section
below is why. Read it before "finishing" this workstream.

## The thing that is easy to get wrong: the two adds do not pair with the same norm

```python
h = self.mix(self.in_ln(x), *ctx)      # norm of the OLD residual
x = x + h                              # add #1
return x + self.mlp(self.post_ln(x))   # add #2
```

- **add #1** (`x = x + h`) is consumed by `post_ln`, three lines later in the same
  method. It folds locally and that is what landed:

  ```python
  normed = self.post_ln.add_residual(x, h, h)
  return x + self.mlp(normed)
  ```

- **add #2** (`x + mlp(...)`) is consumed by the **next layer's `in_ln`** — or, in the
  last layer, by `Qwen35.norm` before the `lm_head` projection. There is no norm after
  it *inside* `Layer.__call__`.

### What breaks if add #2 is folded into `post_ln`

Not a slow result — a **wrong** one. `post_ln` was already applied to `x` *before* add
#2 runs, on a different operand. Folding add #2 back into `post_ln` normalises the sum
`x + mlp(...)` where the layer needs the sum itself, and then the next layer's `in_ln`
normalises an already-normalised value. The residual stream silently stops being a
residual stream: it is still finite and still plausible, which is why this needs saying
rather than testing. Measured on the reference backend: equal is `False`.

Only one residual sum may exist per layer. `add_rms_norm` accumulates into `x` and
normalises into a *separate* buffer precisely so that the sum survives the norm: `x` is
the buffer the next layer adds into. `out` may alias `add` (the mixer output is dead
once the sum has consumed it, so that is where the allocation-free spelling comes from)
but it may never alias `x`. The kernel wrapper rejects `x`-`out` aliasing with
`is_alias_of`, which catches a view as well as the identical tensor.

## Patch B: what it would take

Fold each layer's add #2 into the *following* norm, which means carrying the normalised
stream between loop iterations instead of recomputing it. In `models/qwen35/model.py`,
`_forward` (and only there — `prefill`/`verify`/`batch` each carry their own residual
chain and are not on the 23.5 ms decode path):

```python
normed = self.layers[0].in_ln(x)
for i, layer in enumerate(self.layers):
    normed = layer.forward_fused(x, normed, ctx)   # add #1 into post_ln, returns it
    m = layer.mlp(normed)                          # ... add #2 into the NEXT in_ln
    nxt = self.layers[i + 1].in_ln if i + 1 < len(self.layers) else self.norm
    normed = nxt.add_residual(x, m, m)             # x += m, normed into m
```

Roughly: `Layer` gains one method taking the already-normalised input, and `_forward`
gains the carried buffer plus the loop index. Verified bit-identical to the shipped
spelling over 4 layers — normed chain and residual stream both `True`.

**Why it is deferred anyway.** It restructures `_forward`, and `_forward` is the exact
function `capture()` records. Changing the buffer topology there changes what a captured
graph addresses, and capture on this driver has wedged cards before
(`backends/corex/__init__.py`, `_GRAPH_VERIFIED`). `caps.graph` is `False` by policy so
the eager path is what serves today, but the change still cannot be tested on the card
without risking one. ~0.5 ms of a 23.5 ms step is not worth that, and patch B should be
picked up only alongside a deliberate decision to re-examine capture.

## Numerics: the fold is bit-identical, not merely close

Two details make it so, and neither is optional — both are in `add_rms_norm_kernel`:

1. The sum is narrowed to the activation dtype **before** its square enters the mean
   square. The separate spelling stores the sum in `x` (a bf16 rounding) and the second
   kernel reads that rounded value back, so accumulating the unrounded fp32 sum in one
   pass would compute a different variance. Same reason the gain multiply stays in pass
   2: folding it into pass 1 would change the order of the two roundings.
2. The gain is applied in pass 2 from the value read back out of `x`, exactly as
   `rms_norm_kernel` does.

Checked host-side against the shipped `rms_norm_kernel` driven the same way: identical
bytes over both dtypes, dense and pitched rows, widths 1…2561, M 1…8, and — the case
patch A actually uses — with `out` aliasing `add`. A zero addend reproduces `rms_norm`
byte for byte, so the fused path is the same function as the norm it replaces.

## Aliasing rules, as the kernel enforces them

| relation | allowed | why |
| --- | --- | --- |
| `out` aliases `add` | yes, **dense pitch only** | `add` is consumed in pass 1 before any `out` write; a thread only touches indices it already read |
| `out` aliases `x` | **never** | the residual sum must stay live for the next layer |
| `add` aliases `x` | not checked, not used | would make pass 1 read a partially updated row |

The dense-pitch restriction exists because `add` is addressed by `stride` while `out` is
addressed by `N`: two different pitches over one allocation would let a row's write land
on another row's not-yet-read addend.

Because `out` may alias `add`, the kernel's `add` and `y` pointers are deliberately
**not** `__restrict__` — `__restrict__` would license the compiler to keep a load of
`add` live across the stores to `y`, a miscompile that a single-lane host check cannot
see. `x` and `w` stay restricted.

The decode call path `add_residual(x, h, h)` with 1-D `[hidden]` operands takes the
dense branch: `_rows(h)` is a `[1, 2560]` view sharing `h`'s storage, its row pitch is
2560, so `pitch == N` and the relaxed branch applies. `add_residual` cannot raise on it.

## Reproducing the checks

The host harness used lives in `.tmp_cusyn/` and was removed with the workstream; the
two things worth re-creating if this is revisited, both no-card:

- Compile `norm.cu` as host C++ (`.tmp_cusyn/rewrite.py` + `cuda_stub.h`), call
  `add_rms_norm_kernel` with `blockDim.x = 1` so the shared-tree reduction degenerates
  correctly to one lane, and compare against `rms_norm_kernel` run on the pre-added sum.
  `blockDim.x = 64/128` is sound for the residual (elementwise) but not for the norm.
- `COFFIN_BACKEND=reference python` a layer-shaped chain against both spellings; compare
  with `torch.equal`, not `allclose`. This is what catches a local add-#2 fold, since it
  stays finite and only differs.
