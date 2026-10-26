"""CoreML baseline (sdkv2/sam3.1-coreml @ 9668c49) on this Mac: s/frame for the
backbone and the multiplex-16 tracker step. Baseline only; never shipped.

    tools/baselines/.venv/bin/python -s tools/baselines/coreml_bench.py --inspect
    tools/locked.sh coreml tools/baselines/.venv/bin/python -s tools/baselines/coreml_bench.py

Timing: compute units CPU_AND_GPU (the port's deploy setting; its ANE compile
fails), 2 warm-up calls then N timed calls; report median, min, max. Inputs
are deterministic random tensors of the spec's shapes: CoreML's cost does not
depend on values here (no data-dependent control flow in either graph) [I].
"""

import argparse
import json
import os
import pathlib
import platform
import statistics
import subprocess
import time

import coremltools as ct
import numpy as np

ROOT = pathlib.Path("~/Code/research/models/sam3.1-coreml").expanduser()
BACKBONE = ROOT / "backbone/models/sam3_vision_backbone.mlpackage"
TRACKER = ROOT / "tracker/models/dense_sam3_trackstep.mlpackage"
REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = REPO / "bench/results/baselines"


def describe(path):
    spec = ct.utils.load_spec(str(path))
    d = spec.description
    shape = lambda f: list(f.type.multiArrayType.shape)  # noqa: E731
    dtype = lambda f: ct.proto.FeatureTypes_pb2.ArrayFeatureType.ArrayDataType.Name(f.type.multiArrayType.dataType)  # noqa: E731
    return {
        "inputs": {f.name: {"shape": shape(f), "dtype": dtype(f)} for f in d.input},
        "outputs": {f.name: {"shape": shape(f), "dtype": dtype(f)} for f in d.output},
        "states": {f.name: {"shape": list(f.type.stateType.arrayType.shape)} for f in getattr(d, "state", [])},
    }


def inputs_for(desc, rng):
    np_t = {"FLOAT32": np.float32, "FLOAT16": np.float16, "INT32": np.int32, "DOUBLE": np.float64}
    return {k: rng.standard_normal(v["shape"]).astype(np_t.get(v["dtype"], np.float32)) for k, v in desc["inputs"].items()}


def bench(path, desc, n, rng):
    t0 = time.perf_counter()
    model = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_GPU)
    load_s = time.perf_counter() - t0
    state = model.make_state() if desc["states"] else None
    x = inputs_for(desc, rng)
    call = (lambda: model.predict(x, state)) if state is not None else (lambda: model.predict(x))
    t0 = time.perf_counter()
    call()
    first_s = time.perf_counter() - t0
    call()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        call()
        times.append(time.perf_counter() - t0)
    return {"load_s": round(load_s, 2), "first_call_s": round(first_s, 3), "n": n,
            "median_s": round(statistics.median(times), 4), "min_s": round(min(times), 4), "max_s": round(max(times), 4),
            "all_s": [round(t, 4) for t in times]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("-n", type=int, default=5)
    args = ap.parse_args()
    descs = {"backbone": describe(BACKBONE), "tracker": describe(TRACKER)}
    if args.inspect:
        print(json.dumps(descs, indent=1))
        return
    rng = np.random.default_rng(20261026)
    res = {
        "source": "huggingface.co/sdkv2/sam3.1-coreml @ 9668c4964e216cf9e3a64f0ad812af9d62b33642",
        "compute_units": "CPU_AND_GPU",
        "coremltools": ct.__version__,
        "macos": platform.mac_ver()[0],
        "machine": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
        "load_avg_start": os.getloadavg(),
        "models": descs,
    }
    res["backbone"] = bench(BACKBONE, descs["backbone"], args.n, rng)
    print("backbone", res["backbone"], flush=True)
    res["tracker"] = bench(TRACKER, descs["tracker"], args.n, rng)
    print("tracker", res["tracker"], flush=True)
    res["load_avg_end"] = os.getloadavg()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "coreml.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
