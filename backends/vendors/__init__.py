"""Vendor backend declarations: one module per platform, discovered by filename.

Module: ``coffin/backends/vendors``. A namespace of declarations, not code: each module
here defines one ``SPEC`` and imports nothing but ``backends.vendor``. The loader is
:mod:`coffin.backends.discovery`; the consumer is :mod:`coffin.backends.registry`. A
vendor module may never import ``coffin/ops``, ``models``, ``engine``, ``serving`` or
``cache``.

Drop ``<name>.py`` (or ``<name>/__init__.py`` once it needs a ``kernels/`` directory) in
here, define a module-level ``SPEC`` (:class:`coffin.backends.vendor.VendorSpec`), and
the platform is selectable by ``COFFIN_BACKEND=<name>``. No other file changes.

What a vendor file must satisfy
-------------------------------
* **Import anywhere.** Importing it must work on a host that has never seen the
  hardware: no vendor runtime at module scope, no build, no allocation. The registry
  imports every vendor to ask whether it matches.
* **Declare, do not measure.** Capability values are expectations. A datasheet number
  belongs in ``notes`` until a probe has actually produced it, because a declared flag
  is indistinguishable from a measured one downstream.
* **Start from the reference.** ``build_ops`` may return a partial table; the resolver
  fills every remaining op. A vendor with one kernel is a working backend.
* **Do not claim what is not proven.** Leave ``has_kernels=False`` until kernels run.
  A placeholder is still discovered, still resolvable by name, and still reported by
  ``describe_all`` -- it is just never auto-selected.

The platforms this engine is aimed at
-------------------------------------
They differ in a way the seam has to absorb, which is why detection returns a claim
strength rather than a boolean:

==============  ==============  ================================================
platform        torch space     how a backend recognises it
==============  ==============  ================================================
Hygon DCU       cuda            ROCm build marker (hand-written, backends/hip)
Iluvatar        cuda            device name -- the namespace is shared, so the
                                name is the only discriminator
Enflame         gcu             the namespace itself is unique to the vendor
Moore Threads   musa            the namespace itself is unique to the vendor
==============  ==============  ================================================

Constraints:
    * A vendor module's name is the backend name: ``vendors/foo.py`` is selected by
      ``COFFIN_BACKEND=foo``. Renaming the file renames the backend, so pick the name
      that matches the vendor, not the product line.
    * A leading underscore excludes a module from discovery, which is what an
      in-progress vendor should be called until it is real.
"""
