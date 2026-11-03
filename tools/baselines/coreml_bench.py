"""CoreML baseline (sdkv2/sam3.1-coreml @ 9668c49) on this Mac: s/frame for the
backbone and the multiplex-16 tracker step. Baseline only; never shipped.

    tools/baselines/.venv/bin/python -s tools/baselines/coreml_bench.py --inspect
    tools/locked.sh coreml tools/baselines/.venv/bin/python -s tools/baselines/coreml_bench.py [--units ALL]
    ... coreml_bench.py --only tracker --tracker-precision fp32 -n 32 --out PATH

Timing: compute units CPU_AND_GPU by default (the port's deploy setting; its ANE
compile failed on the port author's machine), or --units ALL / CPU_AND_NE /
CPU_ONLY; 2 warm-up calls then N timed calls; report median, min, max and every
call (`all_s`). The tracker's calls all advance one state, so timed call i is the
state's step i + 2 and `all_s` shows whether a step's cost moves as the memory
bank fills. --tracker-precision fp32 times the port's fp32 package (its precision
reference; the port deploys fp16); --only picks one model; --out names the json. CoreML
falls back per op without saying so, so which unit ran under ALL or CPU_AND_NE
is inferred from the timing, not observed. A model that fails to load or run
under the chosen units is recorded with its error.
Results: bench/results/baselines/coreml.json (CPU_AND_GPU) or coreml-<units>.json. Inputs
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
TRACKERS = {
    "fp16": ROOT / "tracker/models/dense_sam3_trackstep.mlpackage",
    "fp32": ROOT / "tracker/models/dense_sam3_trackstep_fp32.mlpackage",
}
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


def bench(path, desc, n, rng, units):
    t0 = time.perf_counter()
    model = ct.models.MLModel(str(path), compute_units=getattr(ct.ComputeUnit, units))
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
    return {"load_s": round(load_s, 2), "first_call_s": round(first_s, 3), "untimed_calls_before": 2, "n": n,
            "median_s": round(statistics.median(times), 4), "min_s": round(min(times), 4), "max_s": round(max(times), 4),
            "all_s": [round(t, 4) for t in times]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("-n", type=int, default=5)
    ap.add_argument("--units", default="CPU_AND_GPU", choices=["CPU_AND_GPU", "ALL", "CPU_AND_NE", "CPU_ONLY"])
    ap.add_argument("--tracker-precision", default="fp16", choices=sorted(TRACKERS))
    ap.add_argument("--only", default="both", choices=["both", "backbone", "tracker"])
    ap.add_argument("--out", type=pathlib.Path, help="json path (default bench/results/baselines/coreml[-UNITS].json)")
    args = ap.parse_args()
    models = {"backbone": BACKBONE, "tracker": TRACKERS[args.tracker_precision]}
    if args.only != "both":
        models = {args.only: models[args.only]}
    for path in models.values():
        if not path.exists():
            raise SystemExit(f"{path} is missing (fetch it from huggingface.co/sdkv2/sam3.1-coreml)")
    descs = {name: describe(path) for name, path in models.items()}
    if args.inspect:
        print(json.dumps(descs, indent=1))
        return
    rng = np.random.default_rng(20261026)
    res = {
        "source": "huggingface.co/sdkv2/sam3.1-coreml @ 9668c4964e216cf9e3a64f0ad812af9d62b33642",
        "compute_units": args.units,
        "packages": {name: str(path.relative_to(ROOT)) for name, path in models.items()},
        "coremltools": ct.__version__,
        "macos": platform.mac_ver()[0],
        "machine": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
        "load_avg_start": os.getloadavg(),
        "models": descs,
    }
    for name, path in models.items():
        try:
            res[name] = bench(path, descs[name], args.n, rng, args.units)
        except Exception as e:  # recorded, not swallowed: the run's json says this model failed
            res[name] = {"error": f"{type(e).__name__}: {e}"}
        print(name, res[name], flush=True)
    res["load_avg_end"] = os.getloadavg()
    out = args.out or OUT / ("coreml.json" if args.units == "CPU_AND_GPU" else f"coreml-{args.units}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n")
    if any("error" in res[name] for name in models):
        raise SystemExit(f"a model failed; see {out}")


if __name__ == "__main__":
    main()
