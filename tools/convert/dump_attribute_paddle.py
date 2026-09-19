"""Dump PaddleX/PaddleClas attribute ``.pdparams`` into a numpy pickle.

Run this in the **paddlex** environment (needs ``paddle``)::

    python tools/convert/dump_attribute_paddle.py \
        --weights _downloads/official/PP-LCNet_x1_0_pedestrian_attribute_pretrained.pdparams \
        --out     _downloads/official/ped_attr_paddle.pkl

Then convert it with ``tools/convert/convert_attribute_weights.py`` (ptocr env).
"""

import argparse
import pickle

import paddle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sd = paddle.load(args.weights)
    total = sum(int(v.numel()) for v in sd.values())
    print("keys: {} params: {}".format(len(sd), total))
    with open(args.out, "wb") as f:
        pickle.dump({k: v.numpy() for k, v in sd.items()}, f)
    print("saved:", args.out)


if __name__ == "__main__":
    main()
