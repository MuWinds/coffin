"""Resident inference runtime: the model process, its socket and its request loop.

The runtime implementation lives here; process entry points live under
``entrypoints/``.

Module: ``coffin.serving``. Sits at the top of the dependency graph: it drives
``engine`` (requests, scheduling, KV), ``backends`` (the device and its ops) and
``utils`` (tokenizer-side helpers), and nothing imports it except the entry
points. ``runtime_transport`` is the only module that touches the socket, and
``runtime_support`` holds the pieces that must work with no accelerator at all.

Constraints:
    * One engine thread owns the model: connections submit work and drain a
      per-request queue, so no request handler ever runs a forward pass.
    * The platform is never named here. Anything device-shaped -- the namespace,
      the VRAM readout, the timing barrier, whether capture is possible -- comes
      from ``backends``, which is what lets a ``COFFIN_BACKEND=reference`` run
      start on a machine with no card.
"""
