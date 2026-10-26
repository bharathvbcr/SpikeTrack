"""Import-time stand-in so the official SAM 3 code imports on a Mac.

`sam3.model.sam3_tracker_utils` imports `sam3.model.edt` at module level, and
that module imports `triton`, which has no macOS build. (The NMS and
connected-components Triton kernels are imported lazily, only on CUDA.) This
registers a replacement `sam3.model.edt` whose `edt_triton` raises, so a CPU
run that reaches it fails loudly instead of silently taking a different path.
Faking `triton` itself does not work: torch._dynamo probes it on import.
"""

import sys
import types


class TritonUnavailable(RuntimeError):
    pass


def _edt_triton(_data):
    raise TritonUnavailable(
        "sam3.model.edt.edt_triton (a Triton CUDA kernel) was called on a CPU-only reference run"
    )


def install():
    if "sam3.model.edt" in sys.modules:
        return
    edt = types.ModuleType("sam3.model.edt")
    edt.edt_triton = _edt_triton
    sys.modules["sam3.model.edt"] = edt
    _import_multiplex_base_without_cuda()


def _import_multiplex_base_without_cuda():
    """`sam3_multiplex_base` calls torch.cuda.get_device_properties(0) at import
    time only to switch on TF32. Answer that one probe as "not Ampere" while the
    module imports, then restore torch's own function."""
    import torch

    real = torch.cuda.get_device_properties
    torch.cuda.get_device_properties = lambda _i=0: types.SimpleNamespace(major=0, minor=0)
    try:
        import sam3.model.sam3_multiplex_base  # noqa: F401
    finally:
        torch.cuda.get_device_properties = real
