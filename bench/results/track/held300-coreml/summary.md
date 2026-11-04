# Held-out HOTA, first 300 frames, against `default` on today's binary

Baseline reproduces the documented 0.57 (per video {'21': 0.669, '22': 0.669, '36': 0.533, '47': 0.484, '82': 0.529}): yes (0.5700; 21 0.6694, 22 0.6690, 36 0.5330, 47 0.4840, 82 0.5291)

| config | HOTA | dHOTA | 21 d | 22 d | 36 d | 47 d | 82 d | verdict |
|---|---|---|---|---|---|---|---|---|
| fast16 | 0.5705 | +0.0006 | +0.0001 | +0.0001 | +0.0013 | -0.0001 | +0.0000 | PASS (fp16: >= 0.5670, no video < -0.01) |
| t16m16 | 0.5694 | -0.0005 | +0.0001 | +0.0001 | -0.0012 | -0.0001 | -0.0002 | PASS (fp16: >= 0.5670, no video < -0.01) |
