"""Compare torch vs paddle outputs produced by parity_torch_all / parity_paddle_all."""
import glob
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OFFICIAL = os.path.join(ROOT, "_downloads", "official")

print("{:34s} {:>12s} {:>12s} {:>8s}".format("model", "max_abs_diff", "mean_abs_diff", "status"))
worst = 0.0
for t in sorted(glob.glob(os.path.join(OFFICIAL, "*_torch_out.npy"))):
    name = os.path.basename(t)[: -len("_torch_out.npy")]
    p = os.path.join(OFFICIAL, name + "_paddle_out.npy")
    if not os.path.isfile(p):
        continue
    a = np.load(p).astype(np.float64)
    b = np.load(t).astype(np.float64)
    if a.shape != b.shape:
        print("{:34s} SHAPE {}".format(name, (a.shape, b.shape)))
        continue
    d = np.abs(a - b)
    mx, mn = float(d.max()), float(d.mean())
    worst = max(worst, mx)
    status = "OK" if mx < 2e-2 else "CHECK"
    print("{:34s} {:12.3e} {:12.3e} {:>8s}".format(name, mx, mn, status))
print("\nworst max_abs_diff = {:.3e}".format(worst))
