"""Enflame Enflame GCU: the TopsRider / torch_gcu family, through the gcu namespace.

Placeholder -- no kernels, no hardware. Unlike the Iluvatar declaration this one has no
first-hand evidence to lean on: there is no Enflame engine in ``reference/``, so almost
every capability below is a conservative default rather than a measurement. The
asymmetry is worth keeping visible, because a placeholder docstring that reads as
confidently as a measured backend is how an unverified number becomes a design
constraint.

What drove the declaration
--------------------------
* **The namespace is ``gcu``, and it is unique to this vendor.** torch_gcu's guide is
  explicit that moving a tensor to the device means changing ``cuda`` to ``gcu``, and
  that ``torch.cuda.*`` calls become ``torch.gcu.*``. So ``torch.cuda.is_available()``
  does not answer for this card, and detection needs no name marker: the namespace is
  the discriminator.
* **No 64-bit integers.** torch_gcu silently downcasts I64/F64 to 32-bit with a warning,
  and its documentation adds that ``tensor.data_ptr()`` may be wrong for a 64-bit
  tensor. The second half matters beyond indexing: a backend whose addresses cannot be
  trusted cannot support graph capture that replays captured pointers. Hence
  ``index_dtype=torch.int32`` here, which is the one declared field in this file backed
  by documentation rather than inference.
* Everything else -- packed math, graph capture, the extension path, the device name
  string -- is open. The probes that would settle them are listed in
  ``.tmp_dsh/platforms/VENDORS.md``.

Deliberately not declared: a ``device_name`` marker. No public source documents what
``torch.gcu.get_device_name()`` returns, and guessing a substring would create a
detection rule that silently never matches. The namespace already identifies the
vendor, so the honest declaration omits it.

Module: ``coffin/backends/vendors/enflame``. Declarative: imports only
``backends.vendor``, plus ``torch`` for the dtype constant. Must be importable where no
torch_gcu exists -- ``getattr(torch, "gcu", None)`` is how detection avoids assuming
otherwise.

Constraints:
    * ``index_dtype`` is ``int32`` and callers must keep indices in range; silently
      producing wrong positions is the failure mode this field prevents.
    * Do not add a ``weight_layout`` claim without measuring. It is left at the ``"nk"``
      default rather than copied from the Iluvatar finding: the two parts have different
      bandwidth stories and a copied number would be a fiction with a citation.
"""

from __future__ import annotations

import torch

from backends.vendor import Detection, VendorSpec

#: Module-level declaration discovered by :mod:`coffin.backends.discovery`.
#: No ``name_markers``: torch_gcu exposes its own ``gcu`` namespace, so the namespace
#: already identifies the vendor and a name rule would only add a way to fail.
SPEC = VendorSpec(
    name="enflame",
    detection=Detection(namespace="gcu"),
    act_dtype=torch.bfloat16,
    weight_dtype=torch.bfloat16,
    graph=False,
    autotune=False,
    int8_weights=False,
    batch_attn=False,
    attn_split=1,
    # Documented, unlike everything else here: torch_gcu has no I64 and downcasts with a
    # warning, so indices must be declared 32-bit from the start.
    index_dtype=torch.int32,
    priority=20,
    has_kernels=False,
    notes=("namespace gcu (torch.cuda not redirected); no I64 -- indices are int32 and "
           "data_ptr is documented as possibly wrong for 64-bit; packed math, graph "
           "capture, extension path and device-name string all unmeasured"),
)
