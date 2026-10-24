"""Model implementations the engine can serve.

One subpackage per architecture family.  A family provides:

* a model class implementing the ``engine_api.Model`` contract (and the paged /
  hybrid capabilities of ``engine_api.PagedModel`` when its layers carry a
  recurrent state), and
* a ``load_weights`` function with the signature the server uses to keep the
  tensors resident across a reload.

``engine_api`` owns the mapping from a checkpoint config to the subpackage.
"""
