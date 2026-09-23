"""Train on a fixed serialized batch set and print step losses (torch side)."""
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from torchkiln.ocr.losses import build_loss
from torchkiln.ocr.modeling.architectures.base_model import BaseModel
from torchkiln.ocr.optimizer import build_optimizer
from torchkiln.ocr.utils.config import load_config
from torchkiln.ocr.utils.precision import enable_paddle_like_precision

OFF = os.path.join(ROOT, "_downloads", "official")
CFG = os.path.join(ROOT, "configs", "det", "PP-OCRv4_mobile_det.yml")
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 100


def main():
    enable_paddle_like_precision()
    cfg = load_config(CFG)
    dev = torch.device("cuda:0")
    model = BaseModel(cfg["Architecture"]).to(dev)
    st = torch.load(os.path.join(OFF, "PP-OCRv4_mobile_det.pth"), map_location="cpu")
    own = model.state_dict()
    model.load_state_dict(
        {k: v for k, v in st.items() if k in own and tuple(own[k].shape) == tuple(v.shape)},
        strict=False,
    )
    loss_fn = build_loss(cfg["Loss"]).to(dev)
    opt, sched = build_optimizer(cfg["Optimizer"], 20, 25, model)

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
        x = torch.from_numpy(imgs[idx]).to(dev)
        labels = [x] + [torch.from_numpy(m[idx]).to(dev) for m in tmaps]
        ld = loss_fn(model(x), labels)
        opt.zero_grad()
        ld["loss"].backward()
        opt.step()
        sched.step()
        if step % 10 == 0 or step == STEPS - 1:
            out.append("step %d loss %.5f lr %.6f" % (step, float(ld["loss"]), opt.param_groups[0]["lr"]))
    print("\n".join(out))


if __name__ == "__main__":
    main()
