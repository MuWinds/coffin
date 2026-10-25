"""Where does a cached prefill actually spend its time?

``check_paged`` showed a prefix hit reproducing the answer exactly but barely
saving prefill time (1.0x). A prefix hit only pays off if the cost of a prefill
chunk is roughly proportional to its token count, and at these sizes it is not --
this times the pieces so the number can be attributed: one ``model.prefill`` call
per chunk size, the match, the state restore, the reset and the snapshots.

Module: ``coffin/diag``. Times the pieces of a cached prefill through
``prefix_engine`` and ``radix_cache``.

Constraints:
    * The finding this exists for: a correct cache hit still only saves ~1.0x,
      because prefill cost is not proportional to token count at these sizes. The
      attribution here is what says why.
    * Each piece is timed separately (match, restore, reset, snapshots); summing
      kernel rows with torch op rows double counts.
"""
import sys
import time

sys.path.insert(0, "/root/private_data/engine")

import torch
from transformers import AutoTokenizer

from models.qwen35 import Qwen35, load_weights
from engine.prefix_engine import PrefixCacheEngine

MODEL = "/root/private_data/models/Qwen3.5-4B"


def timeit(fn, reps=5) -> float:
    fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / reps * 1000


sd = load_weights(MODEL)
m = Qwen35(MODEL, max_seq=2048, bench=False, weights=sd)
eng = PrefixCacheEngine(m, num_blocks=256, block_size=64)
m.capture()
tok = AutoTokenizer.from_pretrained(MODEL)
body = tok.encode("The history of Paris spans more than two thousand years, beginning "
                  "as a settlement on the banks of the Seine. " * 8)
print(f"body {len(body)} tokens", flush=True)

print("\n--- one model.prefill call, by chunk size ---", flush=True)
for n in (32, 64, 128, 256, 512):
    ids = body[:n]

    def run(ids=ids):
        m.reset()
        m.prefill(ids)

    ms = timeit(run, 5)
    print(f"  n={n:<4} {ms:8.2f} ms   {ms / n * 1000:6.2f} us/token", flush=True)

prompt = body + tok.encode(" When was it founded?")
print(f"\n--- pieces of a cached request ({len(prompt)} tokens) ---", flush=True)
eng.prefill(prompt)                                     # warm the tree
print(f"  reserve          {timeit(lambda: eng.radix.reserve(16), 20):8.3f} ms", flush=True)
print(f"  match_prefix     {timeit(lambda: eng.radix.match_prefix(prompt[:-1], True), 20):8.3f} ms",
      flush=True)
print(f"  model.reset      {timeit(m.reset, 20):8.3f} ms", flush=True)
print(f"  snapshot states  {timeit(m.snapshot_linear_states, 10):8.3f} ms", flush=True)
if m.mtp is not None:
    st = eng.radix.match_prefix(prompt[:-1], True)[2]
    if st is not None:
        print(f"  restore states   {timeit(lambda: m.restore_linear_states(st), 10):8.3f} ms",
              flush=True)

print("\n--- whole request, cold vs warm ---", flush=True)
# A long *unique* prefix, so the first run is genuinely cold and the second can
# skip whole 256-token chunks rather than a few dozen tokens. The cold call is
# timed *once*, before any repetition, because repeating it would warm the tree
# and turn the "cold" measurement into a hit.
uniq = tok.encode("Volume two. The city grew through the centuries, and its walls, "
                  "markets and bridges were rebuilt many times. " * 24)
big = uniq + tok.encode(" Explain the siege of 885 and its aftermath.")
print(f"prompt {len(big)} tokens ({len(big) // 256 + 1} chunks of 256, "
      f"{len(big) // 64 + 1} blocks of 64)", flush=True)
torch.cuda.synchronize()
t0 = time.time()
eng.prefill(big)
torch.cuda.synchronize()
t_cold = (time.time() - t0) * 1000
matched_cold = eng.last_matched
t_warm = timeit(lambda: eng.prefill(big), 5)
print(f"  cold (matched {matched_cold:>4}, once)  {t_cold:8.2f} ms", flush=True)
print(f"  warm (matched {eng.last_matched:>4}, mean of 5)  {t_warm:8.2f} ms   "
      f"({t_cold / t_warm:.2f}x)", flush=True)
print(f"  suffix prefilled: cold {len(big) - matched_cold} tok, "
      f"warm {len(big) - eng.last_matched} tok", flush=True)
