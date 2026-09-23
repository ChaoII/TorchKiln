"""Train on a serialized fixed crop sequence and eval each epoch (torch side)."""
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from torchkiln.ocr.data.simple_dataset import SimpleDataSet
from torchkiln.ocr.losses import build_loss
from torchkiln.ocr.metrics import build_metric
from torchkiln.ocr.modeling.architectures.base_model import BaseModel
from torchkiln.ocr.optimizer import build_optimizer
from torchkiln.ocr.postprocess import build_post_process
from torchkiln.ocr.trainer import det_eval_collate
from torchkiln.ocr.utils.config import load_config
from torchkiln.ocr.utils.precision import enable_paddle_like_precision

OFF = os.path.join(ROOT, "_downloads", "official")
CFG = os.path.join(ROOT, "configs", "det", "PP-OCRv4_mobile_det.yml")
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 5


def main():
    enable_paddle_like_precision()
    cfg = load_config(CFG)
    dev = torch.device("cuda:0")
    m = BaseModel(cfg["Architecture"]).to(dev)
    st = torch.load(os.path.join(OFF, "PP-OCRv4_mobile_det_ptocr.pth"), map_location="cpu")
    own = m.state_dict()
    m.load_state_dict({k: v for k, v in st.items() if k in own and tuple(own[k].shape) == tuple(v.shape)}, strict=False)
    loss_fn = build_loss(cfg["Loss"]).to(dev)
    opt, sched = build_optimizer(cfg["Optimizer"], EPOCHS, 25, m)
    imgs = np.load(os.path.join(OFF, "seq_image.npy"))
    tmaps = [np.load(os.path.join(OFF, "seq_" + k + ".npy")) for k in ["threshold_map", "threshold_mask", "shrink_map", "shrink_mask"]]
    n = imgs.shape[0]
    post = build_post_process(cfg["PostProcess"], global_config=None)
    metric = build_metric(cfg["Metric"])
    ev = DataLoader(SimpleDataSet(cfg, "Eval"), batch_size=1, shuffle=False, num_workers=0, collate_fn=det_eval_collate)
    for ep in range(EPOCHS):
        m.train()
        idxs = np.random.RandomState(ep).permutation(n)
        for b in range(0, n, 8):
            idx = idxs[b : b + 8]
            x = torch.from_numpy(imgs[idx]).to(dev)
            labels = [x] + [torch.from_numpy(t[idx]).to(dev) for t in tmaps]
            ld = loss_fn(m(x), labels)
            opt.zero_grad()
            ld["loss"].backward()
            opt.step()
            sched.step()
        m.eval()
        metric.reset()
        with torch.no_grad():
            for batch in ev:
                if len(batch) == 0:
                    continue
                metric(post(m(batch[0].to(dev)), batch[1].numpy()), batch)
        r = metric.get_metric()
        print("torch epoch %d: %s" % (ep + 1, {k: round(float(v), 4) for k, v in r.items()}))


if __name__ == "__main__":
    main()
