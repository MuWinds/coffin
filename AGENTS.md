# AGENTS.md — coffin comment conventions

Comment contract for this repository. Read before touching any comment.
`coffin/` is its own git repo, so this file is the top-level rule for it.
Scope: every `.py`, `.cu`, `.c`, `.h`, `.cpp` under `coffin/`.

---

## 1. Language: English only

- All comments, docstrings and file headers are written in **English**.
- Do not translate existing English in place. Rewrite it to the standard below.
- Keep verbatim: identifiers, tensor shapes `[B, S, H]`, file paths, URLs, CLI
  flags, units, measured numbers, and tool directives (`# noqa: E402`,
  `# type: ignore[...]`, `# pragma: no cover`). Explain a directive on its own
  line above it, never by rewriting it.
- Non-ASCII text is acceptable only inside quoted sample data (e.g. a prompt
  fragment), never in prose.

---

## 2. Write the Why, not the What

The test: **delete the comment — would a reader now make a wrong change?**
If no, the comment is noise. Delete it.

Delete on sight:

```python
i += 1                  # increment i
for x in xs:            # loop over the list
self.n = 0              # initialise n
return out              # return the result
import torch            # import torch
if not xs:              # check whether empty
```

Keep, because they answer something the code cannot:

```python
# Old API pagination is 1-based; the +1 keeps the two call sites agreeing.
offset = page * size + 1

# gfx906 has 64 KB of LDS, and any dynamic allocation past 48 KB needs an
# opt-in attribute, so the key window is streamed instead of staged whole.
for blk in window:
    ...

# 131072 is the widest row-major-B GEMV that stays correct on this card;
# 196608 was measured returning NaN.
if n > 131072:
    ...

# Materialising the cache as bf16 costs two [nkv, S, hd] copies. Keeping it
# int8 and dequantising in-kernel trades a few ALU ops for 1 GB+ of VRAM.
```

A comment earns its line by answering at least one of:

1. Why must it be written this way, and what breaks otherwise (NaN, 2x VRAM,
   cache aliasing, protocol mismatch)?
2. Which external contract forces it (legacy API, OpenAI wire format,
   checkpoint layout)?
3. Which hardware limit fixes the shape (LDS, bandwidth, wavefront, INT8
   row-major weights)?
4. Where does the magic number come from, and what was measured?
5. What was given up (the trade-off), and why that is acceptable?

---

## 3. Standard docstring format

| Language | Style | Sections |
| --- | --- | --- |
| Python | **Google Style** | `Args:` / `Returns:` / `Raises:` |
| CUDA / C / C++ | **Doxygen** | `@brief` / `@param` / `@return` / `@throws` |

- No Sphinx `:param:` / `:returns:` / `:raises:` (the vendored `reference/`
  package keeps its own style; that is not this repo).
- No NumPy `Parameters` / `----------` sections.
- Section order is fixed: `Args:` then `Returns:` then `Raises:`. Omit a
  section entirely when it would be empty.
- First line is a one-sentence summary ending in a period. Blank line. Then
  the detail and the sections.
- `#:` stays the prefix for constants and dataclass fields.

---

## 4. File-level header

Every source file opens by stating three things.

### Python

```python
"""One-sentence responsibility: why this file exists.

Module: ``coffin/<package>``. Its place in the dependency graph — who calls
it, what it is allowed to call.

Constraints:
    * A hard rule covering the whole file, with the reason attached
      (hardware, precision, invariant, import convention).
    * A second one if it exists.
"""
```

### CUDA / C / C++

```cpp
// @file attn.cu
// @brief Fused single-token full attention over the INT8 KV cache.
//
// Module: coffin/kernels. Loaded by hip_ops.py as a torch extension.
//
// Constraints:
//   * LDS budget is 64 KB per CU; the key window is streamed, never staged.
//   * One block per query head; no cross-block synchronisation.
```

- **Responsibility** is the problem the file solves, not a list of its
  functions.
- **Module** states the dependency direction that applies here (for example
  `ops/` must not import `models/`).
- **Constraints** are file-wide rules, not per-function notes. Typical ones:
  absolute imports because the entry script puts `coffin/` on `sys.path`;
  INT8 row-major weights, so new kernels must accept `[K, N]`; linear-attention
  recurrent state must be snapshotted with the KV or prefix reuse goes wrong;
  no locking, all mutation driven single-threaded by `llm_engine`.

---

## 5. Function / method docstring

Cover the four things: purpose (what and why), inputs, outputs, exceptions.

```python
def match_prefix(self, tokens: list[int], require_states: bool = False) -> tuple[int, int]:
    """Find the longest cached prefix of ``tokens`` in the radix tree.

    Only block-aligned lengths are returned: KV is shared at whole-block
    granularity, and a half-block hit would leave the block table with a
    partial block that copy-on-write cannot refcount.

    Args:
        tokens: Full request tokens (prompt plus generated), must be non-empty.
        require_states: Require a linear-attention state snapshot at the hit
            point. On the hybrid model a KV-only hit would pair shared keys
            with a state that never saw them, so the match is truncated to the
            deepest boundary that has a snapshot.

    Returns:
        ``(prefix_len, node_id)``. ``prefix_len == 0`` means no hit and
        ``node_id`` is then ``-1``.

    Raises:
        ValueError: ``tokens`` is empty. Reject empty requests at the caller;
            returning a 0-length hit here would hide the bug.
    """
```

```cpp
// @brief Single-token full attention over INT8 KV, one pass, online softmax.
//
// @param q        Query vectors, [heads, head_dim], already scaled by 1/sqrt(d).
// @param k_cache  INT8 K view of the paged pool, addressed through the slot map.
// @param out      Output buffer, [heads, head_dim], contiguous.
// @return Nothing. Errors surface through hipGetLastError() on the host side.
// @throws Nothing; HIP reports error codes and illegal arguments trip the
//         host-side assert instead.
```

Detail level:

| Target | Minimum |
| --- | --- |
| Public function with side effects, exceptions, or non-obvious params | All four sections |
| Simple predicate or getter | One-line summary that says why it exists |
| Private helper | Purpose plus its key constraint |
| Test case | No `Args` / `Returns`. State the invariant it guards and why it matters |

`Returns:` describes semantics and edge values, not types — types live in the
signature. `Raises:` states the trigger and what the caller should do about it;
a raise with no explanation is not documentation.

---

## 6. Inline and block comments

1. Non-obvious algorithms, business black magic and magic numbers get a comment
   that names the trade-off. Prefer the measurement over the adjective:
   `3.1 us/token at 64k, ~24 GB/s of KV traffic against a ~600 GB/s roof` beats
   "this is faster".
2. One space after the marker: `# text`, `// text`. Two spaces before an inline
   comment: `x = 1  # why`.
3. A statement-level comment is separated from the code above it by a blank
   line, so it does not read as an aside on the previous statement.
4. A block comment sits flush against the code it explains and describes what
   the block achieves, not each line.
5. Section dividers are plain `# label`. **No ASCII rules** — no `# ----`, no
   `# ====`, no `# ****`. Existing `# -- subsystem ---` lines are removed
   under this rule.
6. No commented-out dead code. If a branch is disabled by a hardware limit,
   say why it is kept and what would re-enable it.

---

## 7. Refactoring discipline

- **Comments only.** No code token, string literal, number, identifier, import
  order or statement may change. The result must be semantically identical.
- Do not rewrite whole large files in one shot. Edit in function-sized batches
  and run the equivalence check after each batch.
- Remove noise, do not add padding. Fewer, sharper comments beat coverage.
- Avoid the stock phrases that mark generated prose: "Note that", "It is
  important to note", "This function is responsible for", "Here we", "In
  order to" (use "to"), "leverage" (use "use"), "utilize", "seamlessly",
  "robust", "comprehensive". State the fact and move on.

---

## 8. Checks

After editing a file:

```bash
cd coffin

# every file still compiles
python -m py_compile $(find . -name '*.py' -not -path './__pycache__/*')

# comments only: AST equal outside docstrings
python diag/check_style.py --ast .

# CUDA/C/C++: token stream equal once comments are stripped
python diag/check_style.py --c-comments .

# conventions: language, spacing, dividers, Sphinx fields, missing sections
python diag/check_style.py --style .
```

`--ast` and `--c-comments` are gates: any reported code change is a bug in the
refactor, restore the file and redo the batch. Do not use `git checkout` to
restore it if other people share the worktree — find the difference and put the
token back by hand.

`--style` splits into two levels. **Errors** are hard: non-English prose, a
missing space after the marker, ASCII dividers, Sphinx field lists, a missing
file header, or no docstring on a public function. **Warnings** are advisory:
missing `Args`/`Returns`/`Raises` sections, a missing `Module:`/`Constraints:`
line, or a docstring missing from a trivial helper. Do not pad a one-line
helper just to clear a warning — that is the coverage-chasing failure mode
section 7 forbids. Treat the warning list as a reading list, not a checklist.

Checklist for one file:

1. Header states responsibility, module and constraints.
2. Public functions have the Google/Doxygen sections, `Args:` covers every
   parameter, `Raises:` names the trigger.
3. Every remaining comment survives the delete-test in section 2.
4. No restating-the-code comments, no non-English prose, no ASCII dividers.
5. `# text` spacing, blank line before statement-level comments.
