"""Train on the same fixed serialized batch set and print step losses (paddle side)."""
import os
import sys

import numpy as np
import paddle
import yaml

PADDLE_REPO = r"E:\PytorchOCR\_downloads\PaddleOCR"
sys.path.insert(0, PADDLE_REPO)

from ppocr.losses import build_loss  # noqa: E402
from ppocr.modeling.architectures.base_model import BaseModel  # noqa: E402
from ppocr.optimizer import build_optimizer  # noqa: E402

ROOT = r"E:\PytorchOCR"
OFF = os.path.join(ROOT, "_downloads", "official")
CFG = os.path.join(ROOT, "configs", "det", "PP-OCRv4_mobile_det.yml")
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 100


def main():
    cfg = yaml.safe_load(open(CFG, encoding="utf-8"))
    model = BaseModel(cfg["Architecture"])
    sd = paddle.load(os.path.join(OFF, "PP-OCRv4_mobile_det_pretrained.pdparams"))
    tgt = model.state_dict()
    for k, v in sd.items():
        v = v.numpy() if hasattr(v, "numpy") else v
        if k in tgt and tuple(tgt[k].shape) == tuple(np.asarray(v).shape):
            tgt[k].set_value(paddle.to_tensor(np.asarray(v)))
    loss_fn = build_loss(cfg["Loss"])
    opt, lr_sched, _ = build_optimizer(cfg["Optimizer"], 20, 25, model)

    imgs = np.load(os.path.join(OFF, "fixed_batch_image.npy"))
    tmaps = [
        np.load(os.path.join(OFF, "fixed_batch_" + k + ".npy"))
        for k in ["threshold_map", "threshold_mask", "shrink_map", "shrink_mask"]
    ]
    bs = 8
    model.train()
    out = []
    for step in range(STEPS):
        idx = [(step * bs + j) % imgs.shape[0] for j in range(bs)]
        x = paddle.to_tensor(imgs[idx])
        labels = [x] + [paddle.to_tensor(m[idx]) for m in tmaps]
        ld = loss_fn(model(x), labels)
        ld["loss"].backward()
        opt.step()
        lr_sched.step()
        opt.clear_grad()
        if step % 10 == 0 or step == STEPS - 1:
            out.append("step %d loss %.5f lr %.6f" % (step, float(ld["loss"]), float(lr_sched.get_lr())))
    print("\n".join(out))


if __name__ == "__main__":
    main()
