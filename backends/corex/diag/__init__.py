"""Acceptance scripts for the CoreX backend: conformance, bandwidth and graph replay.

Nothing here is imported by the backend or the engine. ``diag`` is a directory of
scripts run on the card through ``platform/scow/tools/deploy.py``, next to the
``launch_*.sh`` wrappers that set the cluster's Python and extension cache.

Module: ``coffin/backends/corex/diag``. Depends on ``backends``/``backends.corex``
and ``torch``; it must never be imported from the backend's own code, because a
diagnostic that the runtime depends on is a runtime dependency with a worse name.

Constraints:
    * Every script exits non-zero on a failed check. The job state is the result, so a
      script that prints a failure and exits 0 makes the whole gate meaningless.
    * References are host-side fp64: this card's device fp64 returns zeros and its
      library GEMM returns NaN on the ``lm_head`` width, so neither can be an oracle.
"""
