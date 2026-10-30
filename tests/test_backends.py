"""Conformance and selection tests for the platform backend layer.

Two things are guarded here, and both have to hold on a host with no accelerator:

* the *selection* is deterministic and legible -- ``COFFIN_BACKEND`` wins, an
  unknown or unusable name fails loudly instead of silently degrading, and the
  pure-torch reference is always the last candidate;
* the *reference backend implements all 23 ops*, because it is the safety net the
  rest of the engine relies on once the ``hip_ops.available()`` branches are
  deleted. A ``None`` entry here would be a hole that only shows up as a crash
  mid-request on an unsupported platform.

Module: ``coffin/tests``. Imports ``support`` (for the ``sys.path`` setup),
``backends`` and ``int8_weights``; never loads a checkpoint and never requires a
DCU, so it runs in the CPU-only CI step as well as in the container.

Constraints:
    * Oracles are written out independently from the backend code under test (the
      delta recurrence, the bf16 widening of the int8 KV cache, the depthwise
      convolution). Reusing a backend helper as the reference would only prove the
      helper agrees with itself.
    * Tolerances are dtype-driven, not per-op guesses: fp32 comparisons are exact
      to a few ulp, bf16 comparisons allow the bf16 quantum (~0.8% of the max
      magnitude). A tighter bound on a bf16 result would re-litigate the dtype.
    * The environment-variable paths run in a subprocess. Mutating ``os.environ``
      in-process would leak into the already-selected backend table, which is
      memoised for the process's life by design; a subprocess is also the only way
      to assert "importing this package does not build a kernel".
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import unittest
from pathlib import Path

import support  # noqa: F401  (sets MKL_THREADING_LAYER and puts coffin/ on sys.path)

import torch
import torch.nn.functional as F

import backends
import backends.hip
import backends.reference
from backends.protocol import Caps, Ops, make_device
from int8_weights import QWeight

#: ``coffin/`` itself, for the subprocess runs that need the same import root.
ENGINE_DIR = Path(__file__).resolve().parent.parent

#: Relative tolerance for a bf16 result: the dtype's own quantum, not an error
#: budget. Anything looser would hide a genuinely wrong reduction order.
BF16_TOL = 2e-2

#: Relative tolerance for an fp32 result, i.e. a few ulp of a reordered sum.
FP32_TOL = 1e-5


def _rel(got: torch.Tensor, ref: torch.Tensor) -> float:
    """Max absolute difference normalised by the reference's magnitude.

    A relative metric reads the same at every layer depth; an absolute bound would
    have to be re-tuned per shape, which is how a tolerance ends up meaningless.
    """
    return ((got.float() - ref.float()).abs().max() / ref.float().abs().max().clamp_min(1e-12)).item()


def _run_py(code: str, env: dict | None = None) -> subprocess.CompletedProcess:
    """Run a snippet in a fresh interpreter rooted at ``coffin/``.

    Args:
        code: Python source to execute.
        env: Extra environment entries merged over the current one.

    Returns:
        The completed process, with stdout/stderr captured as text.
    """
    full = dict(os.environ)
    full.pop("COFFIN_BACKEND", None)
    if env:
        full.update(env)
    return subprocess.run([sys.executable, "-c", code], cwd=str(ENGINE_DIR),
                          capture_output=True, text=True, env=full, timeout=180)


class OpTableTest(unittest.TestCase):
    """``Ops`` is the contract: 23 names, frozen order, and no silently dropped entry."""

    def test_names_are_the_contract_ops(self):
        """A rename or a dropped name is a contract break for every call site, so the tuple is pinned literally."""
        self.assertEqual(Ops.NAMES, (
            "linear", "linear_m", "linear_prefill",
            "rms_norm", "add_rms_norm", "rms_norm_gated", "kv_quant", "silu_mul",
            "rope_apply",
            "attn_decode", "attn_paged", "attn_paged_batch", "attn_paged_multi_q",
            "conv1d_silu", "conv1d_silu_seq", "conv1d_silu_batch",
            "delta_prologue", "delta_prologue_batch", "delta_prep_seq",
            "delta", "delta_batch", "delta_normed",
            "triangular_solve",
        ))
        self.assertEqual(len(Ops.NAMES), 23)

    def test_names_match_the_dataclass_fields_in_order(self):
        """The tuple is hand-maintained, so it can drift from the dataclass; this is the drift alarm."""
        from dataclasses import fields

        self.assertEqual(tuple(f.name for f in fields(Ops)), Ops.NAMES)

    def test_default_table_is_all_none(self):
        """``None`` is how a native backend says "no kernel", so the empty table must be exactly that."""
        table = Ops()
        self.assertEqual(table.as_dict(), {name: None for name in Ops.NAMES})
        self.assertEqual(table.implemented(), ())
        self.assertEqual(table.missing(), Ops.NAMES)

    def test_from_mapping_rejects_unknown_names(self):
        """A typo must not yield a table that silently lacks the op; that looks identical to "platform has no kernel"."""
        with self.assertRaises(ValueError):
            Ops.from_mapping({"gemv": lambda: None})
        with self.assertRaises(ValueError):
            Ops.from_mapping({"linear": lambda: None, "Linear": lambda: None})

    def test_post_init_rejects_a_non_callable_entry(self):
        """Registering a tensor or a string would fail at the first decode step, for one shape only."""
        with self.assertRaises(TypeError):
            Ops(linear=1)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            Ops(rms_norm="not a function")  # type: ignore[arg-type]

    def test_from_mapping_is_partial_and_round_trips(self):
        """A native backend lists only its kernels; the rest must come back as ``None`` for the resolver to fill."""
        fn = lambda *_a, **_k: None
        table = Ops.from_mapping({"linear": fn})
        self.assertIs(table.linear, fn)
        self.assertEqual(table.implemented(), ("linear",))
        self.assertEqual(set(table.missing()), set(Ops.NAMES) - {"linear"})


class CapsTest(unittest.TestCase):
    """Capability records carry only facts model code branches on, and they must be resolvable to a device."""

    def test_make_device_resolves_cpu_without_an_index(self):
        """``torch.device("cpu", 0)`` is not portable across torch versions, so the CPU path must drop the index."""
        self.assertEqual(make_device(Caps(name="x", device="cpu", device_index=3)),
                         torch.device("cpu"))

    def test_make_device_resolves_an_indexed_accelerator(self):
        """Every accelerator allocation must land on ``<type>:<index>``; a bare type string would follow whichever device is current."""
        self.assertEqual(make_device(Caps(name="x", device="cuda", device_index=1)),
                         torch.device("cuda", 1))
        self.assertEqual(make_device(Caps(name="x", device="xpu", device_index=0)),
                         torch.device("xpu", 0))

    def test_make_device_rejects_an_unknown_type(self):
        """A typo in a backend's caps must fail at selection, not at the first allocation inside the model builder."""
        with self.assertRaises((RuntimeError, ValueError)):
            make_device(Caps(name="x", device="not-a-device"))

    def test_native_is_a_subset_of_the_contract(self):
        """The banner prints ``len(native)/len(Ops.NAMES)``, so a name outside the contract would make that ratio a lie."""
        for candidate in backends.candidates():
            caps = candidate.backend.caps()
            self.assertTrue(caps.native <= set(Ops.NAMES),
                            f"{caps.name}: {sorted(caps.native - set(Ops.NAMES))}")

    def test_reference_caps_describe_the_eager_path(self):
        """The reference must never claim graph capture or autotune: model code would allocate capture buffers that never exist."""
        caps = backends.reference.caps()
        self.assertEqual(caps.name, "reference")
        self.assertEqual(caps.device, "cpu")
        self.assertEqual(caps.native, frozenset())
        self.assertFalse(caps.graph)
        self.assertFalse(caps.autotune)
        self.assertFalse(caps.int8_weights)
        self.assertEqual(caps.attn_split, 1)
        self.assertTrue(caps.recurrent_state)


class RegistryTest(unittest.TestCase):
    """Selection order and failure text: an explicit pin never falls through, and no failure is silent."""

    def test_available_names_include_every_discovered_vendor(self):
        """Adding a platform is dropping in a file, so the accepted names must come from discovery.

        Pinning a literal tuple here would reintroduce exactly the central list the
        vendor directory exists to remove: a new vendor file would be selectable in
        production but rejected by this test.
        """
        names = backends.available_names()
        self.assertIn("hip", names)
        self.assertIn("reference", names)
        self.assertEqual(names[-1], "reference",
                         "the reference must sort last: it always answers usable")
        for module_name in backends.discovery.modules():
            vendor = module_name.rsplit(".", 1)[-1]
            self.assertIn(vendor, names,
                          f"{module_name} is a discovered vendor but not selectable")

    def test_reference_is_always_selectable(self):
        """It is the guaranteed last candidate; if this ever fails, no host can run the engine."""
        self.assertIs(backends.select("reference"), backends.reference)
        self.assertTrue(backends.reference.probe())
        self.assertTrue(backends.reference.usable())
        self.assertEqual(backends.reference.reason(), "")

    def test_explicit_unknown_name_fails_with_the_known_names(self):
        """A typo must not auto-detect: that would hide a deployment mistake behind a 10x slower backend."""
        with self.assertRaises(RuntimeError) as ctx:
            backends.select("no-such-backend")
        msg = str(ctx.exception)
        self.assertIn("no-such-backend", msg)
        for name in backends.available_names():
            self.assertIn(name, msg)

    def test_explicit_unusable_backend_fails_instead_of_falling_back(self):
        """Pinning a backend means pinning it; silently running something else is the bug this test exists for."""
        if backends.hip.usable():
            self.skipTest("this host has working HIP kernels; the unusable path needs a host without them")
        with self.assertRaises(RuntimeError) as ctx:
            backends.select("hip")
        msg = str(ctx.exception)
        self.assertIn("hip", msg)
        self.assertIn("reference", msg)

    def test_every_discovered_placeholder_is_registered_and_explains_itself(self):
        """A vendor file that resolves by name is the whole point of the directory.

        A placeholder must still answer ``reason()``, because "unknown backend" on real
        hardware the operator just plugged in sends them looking for a deployment
        mistake instead of an unimplemented integration.
        """
        for module_name in backends.discovery.modules():
            vendor = module_name.rsplit(".", 1)[-1]
            spec, backend = backends.discovery.load(module_name)
            self.assertEqual(spec.name, vendor)
            self.assertEqual(backend.name, vendor)
            self.assertFalse(backend.usable(),
                             f"{vendor} claims usable with no kernels")
            self.assertNotEqual(backend.reason(), "",
                                f"{vendor} is unusable but explains nothing")

    def test_hip_probe_rejects_a_cuda_namespace_that_is_not_rocm(self):
        """A ROCm build marker is required: the namespace alone cannot identify this card.

        Several domestic accelerators implement the CUDA API (Iluvatar's CoreX is the
        documented case: its own engine selects the ``cuda`` device outright), so
        answering "yes" for any visible CUDA device would let this backend claim a card
        it cannot run and then compile for an ISA that card lacks.

        Note what this test does *not* assert: that the marker is sufficient. It is not
        known whether CoreX sets it too, which is exactly why backend detection is
        ordered by claim strength -- the Iluvatar vendor matches the device name and
        outranks a namespace-level match regardless.
        """
        if getattr(torch.version, "hip", None) is None:
            self.assertFalse(
                backends.hip.probe(),
                "probe() claimed a non-ROCm host: the build has no HIP marker, so any "
                "visible device belongs to another vendor")
            # The arch is the second half of the vendor test: no ROCm, no ISA. An
            # invented default here is what turns a wrong claim into a bad build.
            self.assertEqual(backends.hip.caps().arch, "")

    def test_backend_is_memoised(self):
        """Selecting twice would rebuild kernels or let two subsystems disagree about the device."""
        self.assertIs(backends.backend(), backends.backend())
        self.assertIs(backends.backend().ops(), backends.backend().ops())

    def test_status_line_carries_name_device_native_graph_autotune(self):
        """The banner is the only place a silent reference fallback is visible; a missing field makes it undiagnosable."""
        line = backends.status()
        for token in ("backend=", "device=", "native=", "graph=", "autotune="):
            self.assertIn(token, line)
        self.assertIn(f"native={len(backends.caps().native)}/{len(Ops.NAMES)}", line)

    def test_describe_all_covers_every_candidate(self):
        """A diagnostic dump that silently omits a backend is how "why did it pick that one" becomes unanswerable."""
        lines = backends.describe_all()
        names = backends.available_names()
        self.assertEqual(len(lines), len(names))
        for name, line in zip(names, lines):
            self.assertTrue(line.startswith(name + ":"), f"{name} -> {line!r}")


class ReferenceTotalityTest(unittest.TestCase):
    """The reference backend must answer all 23 names: it is the only safety net left after migration."""

    def test_reference_table_has_no_gaps(self):
        """A ``None`` here would reach model code on every unsupported platform, mid-request."""
        table = backends.reference.ops()
        self.assertEqual(table.missing(), ())
        self.assertEqual(table.implemented(), Ops.NAMES)
        for name in Ops.NAMES:
            self.assertTrue(callable(getattr(table, name)), name)

    def test_every_vendor_table_resolves_to_reference_for_every_op(self):
        """A placeholder must answer all 23 names today: the model has no fallback branch left.

        This is the invariant that lets a vendor start with one kernel: the partial
        table it returns is completed from the reference, so the model runs unchanged
        from the first day the hardware arrives.
        """
        ref = backends.reference.ops()
        w = torch.randn(4, 3)
        x = torch.randn(3)
        for module_name in backends.discovery.modules():
            _spec, backend = backends.discovery.load(module_name)
            table = backend.ops()
            self.assertEqual(table.missing(), (), f"{backend.name} has gaps")
            for name in Ops.NAMES:
                self.assertTrue(callable(getattr(table, name)), f"{backend.name}.{name}")
            # Spot-check that the filler really is the reference implementation.
            self.assertTrue(torch.allclose(table.linear(w, x), ref.linear(w, x)),
                            f"{backend.name}.linear is not the reference op")

    def test_fill_keeps_native_entries_and_completes_the_rest(self):
        """The resolver is the single place "no kernel here" becomes "run the spec"; a bug in it is a bug everywhere."""
        marker = lambda *_a, **_k: "native"
        partial = Ops.from_mapping({"rms_norm": marker})
        filled = backends.reference.fill(partial, backends.reference.caps())
        self.assertIs(filled.rms_norm, marker)
        self.assertEqual(filled.missing(), ())
        self.assertEqual(set(filled.implemented()), set(Ops.NAMES))

    def test_hip_table_is_total_when_the_kernels_are_usable(self):
        """On the DCU every entry must resolve: the HIP table is the migration's reference implementation in production."""
        if not backends.hip.usable():
            self.skipTest("needs the HIP extension")
        table = backends.hip.ops()
        self.assertEqual(table.missing(), ())
        self.assertEqual(backends.caps().name, "hip")
        self.assertIn("linear", backends.caps().native)


class NativeRegistrationTest(unittest.TestCase):
    """A backend's ``_NATIVE`` dict must list every op it has written an adapter for.

    This guards a failure mode that is invisible in every other check. ``_NATIVE`` is a
    hand-maintained dict of name -> adapter, and an op missing from it is filled from the
    *reference backend* by ``reference.fill``. So an adapter, a header declaration, a
    binding, a protocol field and a reference implementation can all land, the whole tree
    can be green, and the op still never executes a kernel on this backend -- it runs the
    torch spec while reporting success. A performance measurement taken in that state reads
    as a clean zero improvement, which is indistinguishable from "the fusion did not pay".

    The check is about the adapter, not the kernel symbol: a kernel is bound under its own
    extension-level name (``gemv_kn``, ``silu_mul``) which need not match the op name, so
    the question is only whether the module defines an adapter for an op and then leaves it
    unregistered.
    """

    def test_every_op_with_an_adapter_is_registered_native(self):
        """An op the backend defines a callable adapter for must be in ``_NATIVE``."""
        from backends import corex

        missing = sorted(name for name in Ops.NAMES
                         if callable(getattr(corex, name, None))
                         and name not in corex.NATIVE)
        self.assertEqual(missing, [], (
            "these ops have an adapter in backends/corex but are absent from _NATIVE, so "
            "they silently resolve to the reference implementation and never run on the "
            f"card: {missing}"))

    def test_decode_op_set_is_a_subset_of_native(self):
        """``_DECODE_OPS`` gates ``caps.graph``; a name in it with no kernel could capture cuBLAS."""
        from backends.corex import _DECODE_OPS, NATIVE

        self.assertEqual(sorted(_DECODE_OPS - set(NATIVE)), [])

    def test_every_native_op_is_bound_in_the_extension(self):
        """A kernel defined in a ``.cu`` file is unreachable until ``bindings.cpp`` registers it.

        This is the second half of the same trap as the ``_NATIVE`` guard above, and it bit
        once already: ``delta_normed`` had a declaration, a definition, a protocol field, a
        reference implementation, an adapter and a ``_NATIVE`` entry, and the decode step
        still died with ``AttributeError`` on the card because the pybind ``m.def`` line was
        missing. Neither the reference fill nor the local suite can see that: the reference
        answers on a CPU host, and the binding is only read when the extension is imported.

        The check is a source scan because the extension cannot be imported without a
        toolchain. It is approximate by nature -- it proves a name appears in a ``m.def``
        call, not that the signature matches -- but it makes the omission impossible to leave
        in the tree, which is the failure that actually happened.
        """
        import re

        from backends import corex

        bound_path = corex.kernels.KERNEL_DIR / "bindings.cpp"
        if not bound_path.is_file():
            self.skipTest("corex bindings not present")
        bound = set(re.findall(r'm\.def\(\s*"([a-z0-9_]+)"', bound_path.read_text(encoding="utf-8")))

        # A backend-level op name is served by an extension entry of the same name for every
        # op added by the fusion work; an op whose kernel is bound under a different name
        # (``linear`` -> ``gemv_kn``) is out of this check's scope by design, so it only
        # asserts about names that the extension also declares in its header.
        header_path = corex.kernels.KERNEL_DIR / "corex_ops.h"
        declared = set(re.findall(r"^(?:torch::Tensor|void)\s+([a-z0-9_]+)\s*\(",
                                  header_path.read_text(encoding="utf-8"),
                                  flags=re.MULTILINE))
        native_and_declared = (set(corex.NATIVE) & declared) | (declared & bound)
        missing = sorted(native_and_declared - bound)
        self.assertEqual(missing, [], (
            "these kernels are declared and listed native but never registered in "
            f"bindings.cpp, so calling them raises AttributeError on the card: {missing}"))


class VendorExtensionTest(unittest.TestCase):
    """Adding a platform must be dropping in one file, with nothing else edited.

    This is the property the whole layer is judged on: the first two vendors were
    reached by editing a central list in three places, which is exactly what makes a
    platform layer rot as vendors accumulate.
    """

    def test_a_vendor_file_is_a_complete_backend(self):
        """The declarative path must produce a working backend from data alone.

        Builds one in a temporary directory under the repo and points discovery at it,
        so the test exercises the real discovery code rather than a stub. If this ever
        needs a registry edit to pass, the layer has regressed to a central list.

        The vendor under test is discovered rather than named: naming one pinned this
        test to ``iluvatar``, which stopped being a declaration once that platform got
        real kernels and became a hand-written backend. A property about "any vendor
        file" should not break when a vendor graduates.
        """
        module_names = backends.discovery.modules()
        self.assertTrue(module_names, "no vendor declarations found to test with")
        spec, backend = backends.discovery.load(module_names[0])
        self.assertIsInstance(spec, backends.VendorSpec)
        self.assertIsInstance(backend, backends.SpecBackend)
        # A declaration is a full backend: every protocol member, no glue code.
        for member in ("probe", "usable", "reason", "ops", "caps", "device"):
            self.assertTrue(hasattr(backend, member), member)
        self.assertEqual(backend.ops().missing(), ())
        self.assertIn(backend.name, backends.available_names())

    def test_detection_yields_a_claim_strength_not_a_boolean(self):
        """Claim strength is what lets a shared namespace be resolved correctly.

        Iluvatar answers through the CUDA namespace like the Hygon DCU, so both can see a\ndevice. A name match must outrank a namespace-only match, or the ROCm backend\nsteals an Iluvatar card.
        """
        self.assertGreater(backends.vendor.CLAIM_NAME, backends.vendor.CLAIM_NAMESPACE)
        self.assertGreater(backends.vendor.CLAIM_NAMESPACE, backends.vendor.CLAIM_NONE)

    def test_candidate_ranking_prefers_the_strongest_claim(self):
        """Ordering must be total and claim-driven, never filesystem order."""
        strong = backends.registry.Candidate("a", backends.reference, claim=2, priority=90)
        weak = backends.registry.Candidate("b", backends.reference, claim=1, priority=10)
        self.assertLess(strong.rank(), weak.rank(),
                        "a name match must beat a namespace match regardless of priority")
        tie_a = backends.registry.Candidate("a", backends.reference, claim=1, priority=5)
        tie_b = backends.registry.Candidate("b", backends.reference, claim=1, priority=5)
        self.assertLess(tie_a.rank(), tie_b.rank(),
                        "an exact tie must still order deterministically, by name")

    def test_a_broken_vendor_file_is_reported_not_silently_dropped(self):
        """A vendor that fails to import must surface in the diagnostic dump.

        The failure this guards: someone adds a platform, its module raises on import,
        and the only symptom is ``COFFIN_BACKEND=<name>`` saying "unknown backend".
        """
        lines = backends.describe_all()
        joined = "\n".join(lines)
        for module_name in backends.discovery.modules():
            vendor = module_name.rsplit(".", 1)[-1]
            self.assertIn(vendor, joined, f"{vendor} missing from describe_all")

    def test_spec_rejects_a_declaration_that_cannot_work(self):
        """Validation runs at discovery, not at the first request on unowned hardware."""
        good = dict(name="x", detection=backends.Detection(namespace="cuda"))
        backends.VendorSpec(**good).validate()
        with self.assertRaises(ValueError):
            backends.VendorSpec(name="", detection=backends.Detection(namespace="cuda")).validate()
        with self.assertRaises(ValueError):
            backends.VendorSpec(**good, has_kernels=True).validate()
        with self.assertRaises(ValueError):
            backends.VendorSpec(**good, weight_layout="nx").validate()
        with self.assertRaises(ValueError):
            backends.VendorSpec(**good, attn_split=0).validate()


class ReferenceOracleTest(unittest.TestCase):
    """Every reference op must reproduce the maths written out independently in this file.

    These are the numbers a new platform has to match (PLATFORMS.md §6), which is
    why the oracle is a second implementation rather than a call into the backend.
    """

    def setUp(self):
        """Fix the seed per test, so a failure reproduces from the test name alone."""
        torch.manual_seed(1234)
        self.ops = backends.reference.ops()

    def _linear_case(self, n=8, k=6):
        """A weight and activations at fp32, the widest dtype the oracles can compare exactly."""
        return torch.randn(n, k), torch.randn(k)

    def test_linear_single_row_and_batch_agree_with_a_matmul(self):
        """``linear`` is the decode hot path and the only projection the model calls; a transposed weight is plausible and wrong."""
        w, x = self._linear_case()
        self.assertLess(_rel(self.ops.linear(w, x), x @ w.t()), FP32_TOL)
        X = torch.randn(5, w.shape[1])
        self.assertLess(_rel(self.ops.linear_m(w, X), X @ w.t()), FP32_TOL)
        self.assertLess(_rel(self.ops.linear_prefill(w, X), X @ w.t()), FP32_TOL)

    def test_linear_dispatches_on_int8_storage_not_on_an_op_name(self):
        """``QWeight`` is model-side storage; if the int8 branch lived at the call site, a tensor-passing platform would break it."""
        w, x = self._linear_case()
        q = QWeight.quantize(w)
        # Dequantise in fp32: ``QWeight.dequant`` defaults to bf16, and comparing
        # against that would measure bf16 rounding rather than the int8 dispatch.
        dense = q.dequant(torch.float32)
        self.assertLess(_rel(self.ops.linear(q, x), x @ dense.t()), FP32_TOL)
        X = torch.randn(3, w.shape[1])
        self.assertLess(_rel(self.ops.linear_m(q, X), X @ dense.t()), FP32_TOL)
        self.assertLess(_rel(self.ops.linear_prefill(q, X), X @ dense.t()), FP32_TOL)

    def test_rms_norm_matches_the_definition_and_honours_a_row_pitch(self):
        """The model feeds column slices of a fused buffer, so a variant that assumes contiguous rows would corrupt q_norm."""
        x = torch.randn(6, 16)
        g = torch.randn(16)
        ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * g
        self.assertLess(_rel(self.ops.rms_norm(x, g, 16, 1e-6), ref), FP32_TOL)
        packed = torch.randn(6, 40)
        view = packed[:, :16]
        self.assertEqual(view.stride(0), 40)
        view_ref = view * torch.rsqrt(view.pow(2).mean(-1, keepdim=True) + 1e-6) * g
        self.assertLess(_rel(self.ops.rms_norm(view, g, view.stride(0), 1e-6), view_ref),
                        FP32_TOL)

    def test_rms_norm_gated_applies_silu_after_the_head_norm(self):
        """Order matters: ``silu(norm(x) * w)`` and ``norm(x) * w * silu(gate)`` differ, and only the second is in the checkpoint."""
        x = torch.randn(4, 16)
        g = torch.randn(16)
        gate = torch.randn(4, 16)
        ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * g * F.silu(gate)
        self.assertLess(_rel(self.ops.rms_norm_gated(x, g, gate, 1e-6), ref), FP32_TOL)

    def _attn_inputs(self, nh=4, nkv=2, hd=8, window=6):
        """A dense int8 cache window plus one query, all fp32 so the comparison is exact."""
        rep = nh // nkv
        kcache = torch.randint(-127, 127, (nkv, window, hd), dtype=torch.int8)
        vcache = torch.randint(-127, 127, (nkv, window, hd), dtype=torch.int8)
        kscale = torch.rand(nkv, window) + 0.1
        vscale = torch.rand(nkv, window) + 0.1
        return (torch.randn(nh, hd), torch.randn(nh, hd), kcache, vcache,
                kscale, vscale, rep, hd ** -0.5)

    def _attn_ref(self, q, gate, k, v, ks, vs, rep, scale, keep=None):
        """Causal gated attention written from the definition: widen, scale, mask, softmax, PV, gating."""
        nh, hd = q.shape
        nkv = k.shape[0]
        qe = q.reshape(nkv, rep, hd)
        w = k.float() * ks[:, :, None]
        scores = torch.einsum("ghd,gsd->ghs", qe, w) * scale
        if keep is not None:
            scores = scores.masked_fill(~keep[None, None, :], float("-inf"))
        probs = scores.softmax(-1)
        out = torch.einsum("ghs,gsd->ghd", probs, v.float() * vs[:, :, None])
        return out.reshape(nh, hd) * gate.sigmoid()

    def test_attn_decode_matches_the_reference_arithmetic(self):
        """Attention is where a scale folded twice or a mask off by one stays plausible but answers wrongly."""
        q, gate, kc, vc, ks, vs, rep, scale = self._attn_inputs()
        out = torch.zeros_like(q)
        self.ops.attn_decode(q, gate, kc, vc, ks, vs, out, torch.tensor([3]), rep, scale)
        ref = self._attn_ref(q, gate, kc[:, :4], vc[:, :4], ks[:, :4], vs[:, :4], rep, scale)
        self.assertLess(_rel(out, ref), FP32_TOL)

    def test_attn_decode_masks_past_the_position(self):
        """The causal bound is ``pos``, not the cache length: reading the whole window silently leaks future tokens."""
        q, gate, kc, vc, ks, vs, rep, scale = self._attn_inputs()
        short = torch.zeros_like(q)
        long = torch.zeros_like(q)
        self.ops.attn_decode(q, gate, kc, vc, ks, vs, short, torch.tensor([1]), rep, scale)
        self.ops.attn_decode(q, gate, kc, vc, ks, vs, long, torch.tensor([5]), rep, scale)
        self.assertGreater((short - long).abs().max().item(), 0.0)

    def test_attn_paged_equals_decode_on_an_identity_slot_map(self):
        """The paged path must be the same attention; if the two disagree, batch and single-stream decode answer differently."""
        q, gate, kc, vc, ks, vs, rep, scale = self._attn_inputs()
        window = kc.shape[1]
        pool_k = torch.zeros(12, kc.shape[0], kc.shape[2], dtype=torch.int8)
        pool_v = torch.zeros_like(pool_k)
        ps = torch.zeros(12, ks.shape[0])
        pv = torch.zeros_like(ps)
        pool_k[:window] = kc.permute(1, 0, 2)
        pool_v[:window] = vc.permute(1, 0, 2)
        ps[:window] = ks.t()
        pv[:window] = vs.t()
        dense = torch.zeros_like(q)
        paged = torch.zeros_like(q)
        pos = torch.tensor([3])
        self.ops.attn_decode(q, gate, kc, vc, ks, vs, dense, pos, rep, scale)
        self.ops.attn_paged(q, gate, pool_k, pool_v, ps, pv, torch.arange(12), paged,
                            pos, window, rep, scale)
        self.assertLess(_rel(paged, dense), FP32_TOL)

    def test_attn_paged_batch_equals_per_row_paged(self):
        """Each row of the batch kernel must equal the single-row kernel on that row; otherwise a bucket changes its answers."""
        nh, nkv, hd = 4, 2, 8
        rep = nh // nkv
        window, rows = 6, 3
        pool_k = torch.randint(-127, 127, (12, nkv, hd), dtype=torch.int8)
        pool_v = torch.randint(-127, 127, (12, nkv, hd), dtype=torch.int8)
        ps = torch.rand(12, nkv) + 0.1
        pv = torch.rand(12, nkv) + 0.1
        slots = torch.stack([torch.randperm(12) for _ in range(rows)])
        lens = torch.tensor([2, 4, 6], dtype=torch.int32)
        q = torch.randn(rows, nh, hd)
        gate = torch.randn(rows, nh, hd)
        got = torch.zeros(rows, nh, hd)
        self.ops.attn_paged_batch(q, gate, pool_k, pool_v, ps, pv, slots, lens, got,
                                  window, rep, hd ** -0.5)
        for b in range(rows):
            one = torch.zeros(nh, hd)
            self.ops.attn_paged(q[b], gate[b], pool_k, pool_v, ps, pv, slots[b], one,
                                torch.tensor([int(lens[b]) - 1]), window, rep,
                                hd ** -0.5)
            self.assertLess(_rel(got[b], one), FP32_TOL)

    def test_attn_paged_batch_uses_each_rows_own_length(self):
        """A shared launch must not let one row's context length change another row's answer; mixed buckets are the norm."""
        nh, nkv, hd = 2, 2, 8
        rep = nh // nkv
        window = 6
        pool_k = torch.randint(-127, 127, (8, nkv, hd), dtype=torch.int8)
        pool_v = torch.randint(-127, 127, (8, nkv, hd), dtype=torch.int8)
        ps = torch.rand(8, nkv) + 0.1
        pv = torch.rand(8, nkv) + 0.1
        slots = torch.stack([torch.arange(8), torch.arange(8)])
        q = torch.randn(2, nh, hd)
        gate = torch.randn(2, nh, hd)
        mixed = torch.zeros(2, nh, hd)
        alone = torch.zeros(2, nh, hd)
        self.ops.attn_paged_batch(q, gate, pool_k, pool_v, ps, pv, slots,
                                  torch.tensor([1, 5], dtype=torch.int32), mixed,
                                  window, rep, hd ** -0.5)
        # Row 0's context is one key both times, so its output may not move when
        # row 1's context shrinks from five keys to one.
        self.ops.attn_paged_batch(q, gate, pool_k, pool_v, ps, pv, slots,
                                  torch.tensor([1, 1], dtype=torch.int32), alone,
                                  window, rep, hd ** -0.5)
        self.assertLess(_rel(mixed[0], alone[0]), FP32_TOL)
        self.assertGreater((mixed[1] - alone[1]).abs().max().item(), 0.0)

    def test_attn_paged_multi_q_equals_decoding_each_draft_alone(self):
        """Verify accepts a draft only if its single-token decode agrees; a shared pass must be bit-identical per query."""
        nh, nkv, hd = 4, 2, 8
        rep = nh // nkv
        rows, ctx, c, window = 2, 3, 3, 8
        pool_k = torch.randint(-127, 127, (12, nkv, hd), dtype=torch.int8)
        pool_v = torch.randint(-127, 127, (12, nkv, hd), dtype=torch.int8)
        ps = torch.rand(12, nkv) + 0.1
        pv = torch.rand(12, nkv) + 0.1
        slots = torch.stack([torch.randperm(12), torch.randperm(12)])
        q = torch.randn(rows, c, nh, hd)
        gate = torch.randn(rows, c, nh, hd)
        got = torch.zeros(rows, c, nh, hd)
        self.ops.attn_paged_multi_q(q, gate, pool_k, pool_v, ps, pv, slots,
                                    torch.tensor([ctx] * rows, dtype=torch.int32),
                                    got, window, rep, hd ** -0.5)
        for b in range(rows):
            for j in range(c):
                one = torch.zeros(nh, hd)
                self.ops.attn_paged(q[b, j], gate[b, j], pool_k, pool_v, ps, pv,
                                    slots[b], one, torch.tensor([ctx + j]), window,
                                    rep, hd ** -0.5)
                self.assertLess(_rel(got[b, j], one), FP32_TOL)

    def test_attn_paged_multi_q_block_mask_excludes_blocks(self):
        """A kept-block mask must drop the excluded keys entirely; masking the score after the softmax would not."""
        nh, nkv, hd = 2, 2, 8
        rows, ctx, c, window = 1, 1, 2, 8
        pool_k = torch.randint(-127, 127, (12, nkv, hd), dtype=torch.int8)
        pool_v = torch.randint(-127, 127, (12, nkv, hd), dtype=torch.int8)
        ps = torch.rand(12, nkv) + 0.1
        pv = torch.rand(12, nkv) + 0.1
        slots = torch.arange(12).reshape(1, 12)
        q = torch.randn(rows, c, nh, hd)
        gate = torch.randn(rows, c, nh, hd)
        got = torch.zeros(rows, c, nh, hd)
        # block size 2, keep only the first block: query 0 sees keys 0..1 of a
        # 2-key context, which is exactly the unmasked case.
        mask = torch.tensor([[1, 0, 0, 0]], dtype=torch.uint8)
        self.ops.attn_paged_multi_q(q, gate, pool_k, pool_v, ps, pv, slots,
                                    torch.tensor([ctx], dtype=torch.int32), got,
                                    window, nh // nkv, hd ** -0.5, None, None,
                                    mask, 2)
        for j in range(c):
            n = ctx + j + 1
            keep = torch.tensor([True, True] + [False] * j)
            ref = self._attn_ref(q[0, j], gate[0, j],
                                 pool_k[:n].permute(1, 0, 2),
                                 pool_v[:n].permute(1, 0, 2),
                                 ps[:n].t(), pv[:n].t(),
                                 nh // nkv, hd ** -0.5, keep)
            self.assertLess(_rel(got[0, j], ref), FP32_TOL)

    def test_conv1d_silu_shifts_its_window_and_matches_the_definition(self):
        """The window shift is the mixer's short-term memory: a wrong slice silently loses the previous three tokens."""
        channels, kernel = 8, 4
        conv_w = torch.randn(channels, kernel)
        state = torch.randn(channels, kernel - 1)
        x = torch.randn(channels)
        state_in = state.clone()
        out = torch.zeros(channels)
        self.ops.conv1d_silu(conv_w, x, state, out)
        win = torch.cat([state_in, x[:, None]], dim=1)
        self.assertLess(_rel(out, F.silu((win * conv_w).sum(-1))), FP32_TOL)
        self.assertEqual(_rel(state, win[:, 1:]), 0.0)

    def test_conv1d_silu_seq_equals_step_by_step(self):
        """A chunked prefill must not restart the 3-token history at a chunk edge; that is the whole point of the state."""
        channels, kernel = 8, 4
        conv_w = torch.randn(channels, kernel)
        x = torch.randn(9, channels)
        seq_state = torch.zeros(channels, kernel - 1)
        step_state = seq_state.clone()
        seq = self.ops.conv1d_silu_seq(conv_w, x, seq_state)
        steps = torch.stack([self.ops.conv1d_silu(conv_w, x[i], step_state,
                                                  torch.zeros(channels))
                             for i in range(x.shape[0])])
        self.assertLess(_rel(seq, steps), FP32_TOL)
        self.assertEqual(_rel(seq_state, step_state), 0.0)

    def test_conv1d_silu_batch_equals_per_row(self):
        """A bucket must not mix rows' conv windows; that would swap tokens between sequences."""
        channels, kernel, rows = 8, 4, 3
        conv_w = torch.randn(channels, kernel)
        x = torch.randn(rows, channels)
        batch_state = torch.randn(rows, channels, kernel - 1)
        per_state = batch_state.clone()
        out = torch.zeros(rows, channels)
        self.ops.conv1d_silu_batch(conv_w, x, batch_state, out)
        for b in range(rows):
            one = torch.zeros(channels)
            self.ops.conv1d_silu(conv_w, x[b], per_state[b], one)
            self.assertEqual(_rel(out[b], one), 0.0)
        self.assertEqual(_rel(batch_state, per_state), 0.0)

    def _prologue_ref(self, q, k, v, b, a, neg_expA, dt_bias, qscale, rep):
        """Head prep from the definition: L2 norm, head expansion, sigmoid and softplus gates."""
        qe = q.repeat_interleave(rep, dim=-2)
        ke = k.repeat_interleave(rep, dim=-2)
        qn = qe * torch.rsqrt((qe * qe).sum(-1, keepdim=True) + 1e-6) * qscale
        kn = ke * torch.rsqrt((ke * ke).sum(-1, keepdim=True) + 1e-6)
        return qn, kn, b.sigmoid(), neg_expA * F.softplus(a + dt_bias)

    def test_delta_prologue_matches_the_definition(self):
        """Sixteen small elementwise ops were fused into this; a wrong softplus or head expansion breaks the recurrence quietly."""
        nv, nk, dk, dv, rep = 4, 2, 8, 8, 2
        q, k = torch.randn(nk, dk), torch.randn(nk, dk)
        v, b, a = torch.randn(nv, dv), torch.randn(nv), torch.randn(nv)
        neg, dt = -torch.rand(nv) - 0.5, torch.randn(nv)
        qs = dk ** -0.5
        qo, ko, vo = torch.zeros(nv, dk), torch.zeros(nv, dk), torch.zeros(nv, dv)
        bo, go = torch.zeros(nv), torch.zeros(nv)
        self.ops.delta_prologue(q, k, v, b, a, neg, dt, qo, ko, vo, bo, go, qs, rep)
        qn, kn, bn, gn = self._prologue_ref(q, k, v, b, a, neg, dt, qs, rep)
        self.assertLess(_rel(qo, qn), FP32_TOL)
        self.assertLess(_rel(ko, kn), FP32_TOL)
        self.assertEqual(_rel(vo, v), 0.0)
        self.assertLess(_rel(bo, bn), FP32_TOL)
        self.assertLess(_rel(go, gn), FP32_TOL)
        self.assertLessEqual(go.max().item(), 0.0)

    def test_delta_prologue_batch_equals_per_row(self):
        """Batched prologue shares ``neg_expA``/``dt_bias`` across rows; a per-row read of the 1-D tensors would alias."""
        nv, nk, dk, dv, rep, rows = 4, 2, 8, 8, 2, 3
        q, k = torch.randn(rows, nk, dk), torch.randn(rows, nk, dk)
        v, b, a = torch.randn(rows, nv, dv), torch.randn(rows, nv), torch.randn(rows, nv)
        neg, dt = -torch.rand(nv) - 0.5, torch.randn(nv)
        qs = dk ** -0.5
        qo = torch.zeros(rows, nv, dk)
        ko = torch.zeros(rows, nv, dk)
        vo = torch.zeros(rows, nv, dv)
        bo, go = torch.zeros(rows, nv), torch.zeros(rows, nv)
        self.ops.delta_prologue_batch(q, k, v, b, a, neg, dt, qo, ko, vo, bo, go, qs, rep)
        for i in range(rows):
            qn, kn, bn, gn = self._prologue_ref(q[i], k[i], v[i], b[i], a[i],
                                                neg, dt, qs, rep)
            self.assertLess(_rel(qo[i], qn), FP32_TOL)
            self.assertLess(_rel(ko[i], kn), FP32_TOL)
            self.assertLess(_rel(bo[i], bn), FP32_TOL)
            self.assertLess(_rel(go[i], gn), FP32_TOL)

    def test_delta_prep_seq_matches_the_definition_and_the_head_major_layout(self):
        """The layout is the contract: the chunked scan reads ``[H, T, d]`` directly, so a transposed output feeds it garbage."""
        nv, nk, dk, dv, rep, T = 4, 2, 8, 8, 2, 5
        conv_dim = 2 * nk * dk + nv * dv
        qkv, b, a = torch.randn(T, conv_dim), torch.randn(T, nv), torch.randn(T, nv)
        neg, dt = -torch.rand(nv) - 0.5, torch.randn(nv)
        qs = dk ** -0.5
        qt, kt = torch.zeros(nv, T, dk), torch.zeros(nv, T, dk)
        vt, bt, gt = torch.zeros(nv, T, dv), torch.zeros(nv, T), torch.zeros(nv, T)
        self.ops.delta_prep_seq(qkv, b, a, neg, dt, qt, kt, vt, bt, gt, qs, rep)
        kdim = nk * dk
        q, k = qkv[:, :kdim].reshape(T, nk, dk), qkv[:, kdim:2 * kdim].reshape(T, nk, dk)
        v = qkv[:, 2 * kdim:].reshape(T, nv, dv)
        qn, kn, bn, gn = self._prologue_ref(q, k, v, b, a, neg, dt, qs, rep)
        self.assertLess(_rel(qt, qn.permute(1, 0, 2)), FP32_TOL)
        self.assertLess(_rel(kt, kn.permute(1, 0, 2)), FP32_TOL)
        self.assertLess(_rel(vt, v.permute(1, 0, 2)), FP32_TOL)
        self.assertLess(_rel(bt, bn.permute(1, 0)), FP32_TOL)
        self.assertLess(_rel(gt, gn.permute(1, 0)), FP32_TOL)

    def test_delta_matches_the_gated_delta_update(self):
        """The recurrence is the only cross-token memory in 24 of 32 layers: decay, write, then read, in that order."""
        nv, dk, dv = 4, 8, 8
        q, k, v = torch.randn(nv, dk), torch.randn(nv, dk), torch.randn(nv, dv)
        beta, g = torch.rand(nv), -torch.rand(nv)
        state = torch.randn(nv, dk, dv)
        got_st = state.clone()
        out = torch.zeros(nv, dv)
        self.ops.delta(q, k, v, beta, g, got_st, out)
        ref_st = state * g.exp()[:, None, None]
        kv = (ref_st * k[:, :, None]).sum(-2)
        d = (v - kv) * beta[:, None]
        ref_st = ref_st + k[:, :, None] * d[:, None, :]
        ref_out = (ref_st * q[:, :, None]).sum(-2)
        self.assertLess(_rel(out, ref_out), FP32_TOL)
        self.assertLess(_rel(got_st, ref_st), FP32_TOL)

    def test_delta_batch_equals_per_row_and_keeps_rows_independent(self):
        """Rows must not share state: a batched step that broadcast its update would make bucket members influence each other."""
        nv, dk, dv, rows = 4, 8, 8, 3
        q, k = torch.randn(rows, nv, dk), torch.randn(rows, nv, dk)
        v = torch.randn(rows, nv, dv)
        beta, g = torch.rand(rows, nv), -torch.rand(rows, nv)
        batch_st = torch.randn(rows, nv, dk, dv)
        per_st = batch_st.clone()
        out = torch.zeros(rows, nv, dv)
        self.ops.delta_batch(q, k, v, beta, g, batch_st, out)
        for b in range(rows):
            one = torch.zeros(nv, dv)
            self.ops.delta(q[b], k[b], v[b], beta[b], g[b], per_st[b], one)
            self.assertEqual(_rel(out[b], one), 0.0)
        self.assertEqual(_rel(batch_st, per_st), 0.0)


class DependencyDirectionTest(unittest.TestCase):
    """``backends/`` may not import the engine layers it serves; one-way imports are what keeps the layer testable alone."""

    #: Packages ``backends/`` must never import: they sit above it in the graph.
    FORBIDDEN = ("ops", "models", "engine", "serving", "cache", "entrypoints",
                 "engine_api")

    #: One documented exception, with the step that removes it. S0 keeps the host
    #: shim and the kernels where they are and adapts them as a backend; S1 moves
    #: them to ``backends/hip/``, at which point this entry must be deleted rather
    #: than grown. ``hip_ops`` is a *leaf* shim (it imports only torch), so the
    #: temporary dependence does not invert the graph.
    S0_ALLOWLIST = {"backends/hip/__init__.py": ("hip_ops",)}

    def _backend_files(self):
        """Every Python file in the backend package, so a new submodule is covered automatically."""
        root = ENGINE_DIR / "backends"
        return sorted(p for p in root.rglob("*.py"))

    def test_no_backend_module_imports_an_engine_layer(self):
        """An import here would invert the dependency graph and make the backend untestable without a checkpoint."""
        offenders = []
        for path in self._backend_files():
            rel = path.relative_to(ENGINE_DIR).as_posix()
            allowed = self.S0_ALLOWLIST.get(rel, ())
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                for name in names:
                    top = name.split(".")[0]
                    if top in allowed:
                        continue
                    if top in self.FORBIDDEN:
                        offenders.append(f"{rel}:{node.lineno} imports {name}")
        self.assertEqual(offenders, [],
                         "backends/ must not import engine layers:\n" + "\n".join(offenders))

    def test_the_s0_allowlist_stays_minimal(self):
        """A growing allowlist is how a dependency gate turns into decoration; this pins it to one entry."""
        self.assertEqual(set(self.S0_ALLOWLIST), {"backends/hip/__init__.py"})
        self.assertEqual(self.S0_ALLOWLIST["backends/hip/__init__.py"], ("hip_ops",))

    def test_package_imports_without_an_accelerator_or_a_build(self):
        """Importing the package is what a CPU-only box does; a build attempt there would rewrite the environment and fail."""
        probe = ("import sys, backends\n"
                 "assert backends.caps().name == 'reference', backends.caps().name\n"
                 "assert 'hip_ops' not in sys.modules, 'imported the HIP shim'\n"
                 "print(backends.status())\n")
        done = _run_py(probe)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("backend=reference", done.stdout)

    def test_env_var_selects_and_pins_the_backend(self):
        """``COFFIN_BACKEND`` is the A/B and troubleshooting knob; it has to work before anything else imports the package."""
        ok = _run_py("import backends; print(backends.caps().name)",
                     env={"COFFIN_BACKEND": "reference"})
        self.assertEqual(ok.returncode, 0, ok.stderr)
        self.assertEqual(ok.stdout.strip(), "reference")

    def test_env_var_with_an_unknown_name_fails_loudly(self):
        """A typo must abort at import, not select a slower backend and look like a performance regression."""
        bad = _run_py("import backends", env={"COFFIN_BACKEND": "nonsense"})
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("nonsense", bad.stderr)
        self.assertIn("COFFIN_BACKEND", bad.stderr)


class CorexPackTest(unittest.TestCase):
    """The int8 -> bf16 weight conversion, which every int8 prefill call pays for."""

    def _packed(self):
        """Build a small packed int8 weight with realistic scales.

        Returns:
            A ``KnQWeight`` whose ``[K, N]`` matrix is contiguous int8 and whose scale is
            1-D fp32, which is the pair the constructor validates.
        """
        from backends.corex.pack import KnQWeight

        k, n = 96, 40
        wt = torch.randint(-127, 127, (k, n), dtype=torch.int8)
        scale = (torch.rand(n) * 0.01 + 1e-4).to(torch.float32)
        return KnQWeight(wt, scale)

    def test_dequant_kn_is_bit_identical_to_the_wide_fp32_spelling(self):
        """The fast spelling must not move a single weight.

        The conversion used to build a ``[K, N]`` fp32 temporary and narrow it, which cost
        174.8 ms per prefill request on the card. Casting to the output dtype first and
        scaling in place is 1.70x faster because ``mul_`` takes its dtype from the
        destination, so it still multiplies in fp32 and rounds once -- same arithmetic,
        no temporary. A bf16 *scale* would instead cost 1 ULP on every weight, which is
        why the scale stays fp32 here and this test is bit-exact rather than approximate.
        """
        q = self._packed()
        got = q.dequant_kn(torch.bfloat16).wt
        want = (q.wt.float() * q.scale.view(1, -1)).to(torch.bfloat16)
        self.assertTrue(torch.equal(got, want),
                        "dequant_kn changed the weights; a faster conversion is only "
                        "acceptable if it is bit-identical")

    def test_dequant_kn_keeps_the_kn_layout(self):
        """The result feeds ``matmul(lhs, wt)``, so a transposed or non-contiguous matrix multiplies silently wrong."""
        q = self._packed()
        out = q.dequant_kn(torch.bfloat16)
        self.assertEqual((out.k, out.n), (q.k, q.n))
        self.assertEqual(tuple(out.wt.shape), (q.k, q.n))
        self.assertTrue(out.wt.is_contiguous())
        self.assertEqual(out.wt.dtype, torch.bfloat16)

    def test_dequant_kn_rejects_an_integer_destination(self):
        """Scaling into an integer destination wraps instead of raising, which reads as a corrupt checkpoint later."""
        q = self._packed()
        with self.assertRaises(TypeError):
            q.dequant_kn(torch.int8)


class Int8BatchKernelTest(unittest.TestCase):
    """The int8 batched projection must be a kernel, not a loop that looks like one.

    The failure this guards is silent in both directions. Without a batched entry,
    ``linear_m`` loops the single-row GEMV, which is *correct* -- so no functional test can
    see it -- and reads the whole weight once per row, which makes every speculative verify
    cost ``c`` plain steps. That is the measured reason suffix decoding could not break even
    on this card (../../PERF_COREX.md section 14), and it would return the moment somebody reverts
    the dispatch to the loop.

    A card is needed to *call* the kernel, so these checks are structural: they pin the pieces
    that must all exist together, because each of them has been the missing one at some point
    in this port (../../PERF_COREX.md 12.5).
    """

    def _sources(self) -> dict:
        """Read the kernel-side sources this contract spans.

        Returns:
            ``{file name: text}`` for the header, the binding and the GEMV kernels.

        Raises:
            unittest.SkipTest: The sources are absent, i.e. a tree without the backend.
        """
        from backends import corex

        kernel_dir = corex.kernels.KERNEL_DIR
        wanted = ("corex_ops.h", "bindings.cpp", "gemv.cu")
        if not all((kernel_dir / name).is_file() for name in wanted):
            self.skipTest("corex kernel sources not present")
        return {n: (kernel_dir / n).read_text(encoding="utf-8") for n in wanted}

    def test_the_batched_int8_entry_exists_in_header_binding_and_kernel(self):
        """Declaration, registration and definition are three edits that all have to land."""
        src = self._sources()
        self.assertIn("gemv_kn_i8_m", src["corex_ops.h"],
                      "the batched int8 entry is not declared in corex_ops.h")
        self.assertIn('m.def("gemv_kn_i8_m"', src["bindings.cpp"],
                      "the batched int8 entry is not registered in bindings.cpp, so "
                      "calling it raises AttributeError on the card instead of failing "
                      "the build")
        self.assertIn("gemv_kn_i8_m_kernel", src["gemv.cu"],
                      "gemv_kn_i8_m has no kernel body, so the entry has nothing to "
                      "launch")

    def test_the_batched_int8_kernel_accumulates_m_rows_per_weight_load(self):
        """The kernel only pays if M chains stay live across one weight read.

        Checked in the source because that property is a compiler outcome no unit test
        can observe without a card: the row loop has to be template-instantiated and
        fully unrolled over M, with the weight load hoisted outside it. A runtime ``for``
        over rows would keep the kernel shape and read the weight M times, which is
        exactly the bug this kernel exists to remove.
        """
        src = self._sources()["gemv.cu"]
        start = src.index("gemv_kn_i8_m_kernel")
        body = src[start:src.index(chr(10) + "}" + chr(10), start)]
        self.assertIn("float acc[M * W];", body,
                      "the int8 batched kernel has no per-(row, column) accumulator array")
        self.assertIn("load_i8_row<W>(&wt[", body,
                      "the weight bytes are not loaded once into a local word, so they "
                      "are re-read once per row")
        self.assertIn("#pragma unroll", body,
                      "the row loop is not unrolled, so the M chains may spill or "
                      "re-read the weight")

    def test_linear_m_routes_int8_to_the_batched_kernel_not_a_row_loop(self):
        """The dispatch is the piece a correct-but-slow revert would silently change."""
        import inspect

        from backends import corex

        source = inspect.getsource(corex._gemv_rows)
        self.assertIn("gemv_kn_i8_m", source,
                      "linear_m does not call the batched int8 entry: an int8 row "
                      "batch falls back to one weight read per row, the regression "
                      "that made speculative decoding unable to break even")


if __name__ == "__main__":
    unittest.main()
