"""Dump every official .pdparams in _downloads/official to a numpy pickle.

Run with the Paddle interpreter (the `paddlex` conda env).
"""
import glob
import os
import pickle

import numpy as np
import paddle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OFFICIAL = os.path.join(ROOT, "_downloads", "official")


def main():
    for src in sorted(glob.glob(os.path.join(OFFICIAL, "*.pdparams"))):
        name = os.path.splitext(os.path.basename(src))[0]
        dst = os.path.join(OFFICIAL, name + ".pkl")
        sd = paddle.load(src)
        if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]
        out = {}
        for k, v in sd.items():
            if hasattr(v, "numpy"):
                v = v.numpy()
            out[k] = np.asarray(v)
        with open(dst, "wb") as f:
            pickle.dump(out, f, protocol=4)
        print("dumped {:40s} {} tensors".format(name, len(out)))


if __name__ == "__main__":
    main()
