"""Moore Threads Moore Threads: the MUSA family, through the musa namespace.

Placeholder -- no kernels, no hardware. Included alongside Iluvatar and Enflame because it
is the third of the domestic accelerators this engine is aimed at, and because its
detection story is the clearest of the three, which makes it a good reference for how a
new vendor should declare itself.

What drove the declaration
--------------------------
* **The namespace is ``musa``, and ``torch.cuda`` is deliberately not redirected.**
  torchada's documentation states outright that ``torch.cuda.is_available()`` returns
  ``False`` on a MUSA build so that platform detection works normally. That is the
  ideal case for this layer: no shared-namespace ambiguity, no name rule needed, the
  namespace alone identifies the vendor.
* **The device name carries the part.** ``musaInfo`` reports names like
  ``MTT S4000``, and downstream projects match on the ``MTT`` prefix. A name marker is
  still declared even though the namespace already decides, because it costs nothing
  and it makes the declaration robust if the vendor ever ships a compat layer that does
  redirect ``cuda`` -- at which point the name is the only thing left.

Everything else is open: packed math, graph capture, the extension path, and whether an
int8 weight path pays off. The probes that settle them are in
``.tmp_dsh/platforms/VENDORS.md``.

Module: ``coffin/backends/vendors/moorethreads``. Declarative: imports only
``backends.vendor``, plus ``torch`` for the dtype constant. Must be importable where no
MUSA build exists.

Constraints:
    * ``name_markers`` are substrings matched case-insensitively. ``"mtt"`` is chosen
      over the full product name because the part number changes per generation and the
      prefix has not.
    * Do not declare a capability this file cannot justify; see the sibling vendor
      modules for the same rule stated at length.
"""

from __future__ import annotations

import torch

from backends.vendor import Detection, VendorSpec

#: Module-level declaration discovered by :mod:`coffin.backends.discovery`.
#: The name marker is redundant while ``torch.cuda`` stays un-redirected, and becomes
#: the only working discriminator if a compat layer ever changes that. Cheap insurance.
SPEC = VendorSpec(
    name="moorethreads",
    detection=Detection(namespace="musa", name_markers=("mtt", "moore")),
    act_dtype=torch.bfloat16,
    weight_dtype=torch.bfloat16,
    graph=False,
    autotune=False,
    int8_weights=False,
    batch_attn=False,
    attn_split=1,
    priority=30,
    has_kernels=False,
    notes=("namespace musa, torch.cuda intentionally not redirected "
           "(is_available() is False here); device name reports MTT parts; packed "
           "math, graph capture and extension path unmeasured"),
)
