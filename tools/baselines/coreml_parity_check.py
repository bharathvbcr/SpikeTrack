"""Does the CoreML backbone compute what upstream's trunk + propagation neck computes?

    tools/reference/.venv/bin/python -s -c "from safetensors.numpy import load_file; import numpy as np; f = load_file('fixtures/backbone/visem_22_f0.safetensors'); np.save('/tmp/cm_img.npy', f['input.image']); np.save('/tmp/cm_want.npy', f['neck.propagation_convs.2.out'])"
    tools/baselines/.venv/bin/python -s tools/baselines/coreml_parity_check.py /tmp/cm ~/Code/research/models/sam3.1-coreml/backbone/models/sam3_vision_backbone.mlpackage

(The baselines venv has coremltools but not safetensors, so the fixture's two tensors go
through .npy files written by the reference venv; argv[1] is their common prefix.)

Feeds the backbone fixture's upstream input to the CoreML backbone and compares vis72 with
upstream's neck.propagation_convs.2.out (both 72x72x256). Tries the input as the fixture
stores it (normalized, mean/std 0.5) and as [0, 1] in case the port normalizes inside."""

import sys

import coremltools as ct
import numpy as np

fx_path, pkg = sys.argv[1], sys.argv[2]
img = np.load(fx_path + "_img.npy")
want = np.load(fx_path + "_want.npy").astype(np.float64).reshape(256, 72 * 72)
model = ct.models.MLModel(pkg, compute_units=ct.ComputeUnit.CPU_AND_GPU)


def cos(a, b):
    a, b = a.ravel(), b.ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


for label, x in [("normalized input", img), ("[0,1] input", img * 0.5 + 0.5)]:
    out = model.predict({"image": x})
    vis = np.asarray(out["vis72"], dtype=np.float64).reshape(72 * 72, 256).T  # [5184,1,256] -> [256, 5184]
    print(f"{label}: vis72 vs upstream propagation neck level 2: cosine {cos(vis, want):.6f}, max abs {np.abs(vis - want).max():.3e}, ref max {np.abs(want).max():.3e}")
