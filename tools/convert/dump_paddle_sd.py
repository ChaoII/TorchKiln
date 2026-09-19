"""Dump a Paddle .pdparams state dict to a pickle of numpy arrays.

Run this with the *Paddle* interpreter (e.g. the `paddlex` conda env), so that
no Paddle installation is needed in the PyTorch environment.

Usage:
    python tools/convert/dump_paddle_sd.py <in.pdparams> <out.pkl>
"""
import pickle
import sys

import numpy as np
import paddle


def main():
    src, dst = sys.argv[1], sys.argv[2]
    sd = paddle.load(src)
    if not isinstance(sd, dict):
        raise TypeError("expected a state dict, got {}".format(type(sd)))

    out = {}
    for k, v in sd.items():
        if hasattr(v, "numpy"):
            v = v.numpy()
        out[k] = np.asarray(v)

    with open(dst, "wb") as f:
        pickle.dump(out, f, protocol=4)
    print("dumped {} tensors -> {}".format(len(out), dst))


if __name__ == "__main__":
    main()
