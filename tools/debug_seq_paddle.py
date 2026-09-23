"""Train on the SAME serialized fixed crop sequence and eval each epoch (paddle side)."""
import logging
import os
import sys

import numpy as np
import paddle
import yaml

PADDLE_REPO = r"E:\TorchKiln\_downloads\PaddleOCR"
sys.path.insert(0, PADDLE_REPO)

from ppocr.data import build_dataloader  # noqa: E402
from ppocr.losses import build_loss  # noqa: E402
from ppocr.metrics import build_metric  # noqa: E402
from ppocr.modeling.architectures.base_model import BaseModel  # noqa: E402
from ppocr.optimizer import build_optimizer  # noqa: E402
from ppocr.postprocess import build_post_process  # noqa: E402

ROOT = r"E:\TorchKiln"
OFF = os.path.join(ROOT, "_downloads", "official")
CFG = os.path.join(ROOT, "configs", "det", "PP-OCRv4_mobile_det.yml")
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 5


def main():
    cfg = yaml.safe_load(open(CFG, encoding="utf-8"))
    D = os.path.join(ROOT, "datasets", "ocr_det_dataset_examples").replace("\\", "/")
    m = BaseModel(cfg["Architecture"])
    sd = paddle.load(os.path.join(OFF, "PP-OCRv4_mobile_det_pretrained.pdparams"))
    tgt = m.state_dict()
    for k, v in sd.items():
        v = v.numpy() if hasattr(v, "numpy") else v
        if k in tgt and tuple(tgt[k].shape) == tuple(np.asarray(v).shape):
            tgt[k].set_value(paddle.to_tensor(np.asarray(v)))
    loss_fn = build_loss(cfg["Loss"])
    opt, lr, _ = build_optimizer(cfg["Optimizer"], EPOCHS, 25, m)

    imgs = np.load(os.path.join(OFF, "seq_image.npy"))
    tmaps = [np.load(os.path.join(OFF, "seq_" + k + ".npy")) for k in ["threshold_map", "threshold_mask", "shrink_map", "shrink_mask"]]
    n = imgs.shape[0]

    post = build_post_process(cfg["PostProcess"], cfg.get("Global"))
    metric = build_metric(cfg["Metric"])
    cfg["Eval"]["dataset"]["data_dir"] = D
    cfg["Eval"]["dataset"]["label_file_list"] = [D + "/val.txt"]
    cfg["Eval"]["loader"]["num_workers"] = 0
    lg = logging.getLogger("x")
    lg.setLevel(logging.ERROR)
    dl = build_dataloader(cfg, "Eval", paddle.device.get_device(), lg, None)

    for ep in range(EPOCHS):
        m.train()
        idxs = np.random.RandomState(ep).permutation(n)
        for b in range(0, n, 8):
            idx = idxs[b : b + 8]
            x = paddle.to_tensor(imgs[idx])
            labels = [x] + [paddle.to_tensor(t[idx]) for t in tmaps]
            ld = loss_fn(m(x), labels)
            ld["loss"].backward()
            opt.step()
            lr.step()
            opt.clear_grad()
        m.eval()
        metric.reset()
        with paddle.no_grad():
            for batch in dl:
                nb = [b.numpy() for b in batch]
                metric(post(m(batch[0]), nb[1]), nb)
        r = metric.get_metric()
        print("paddle epoch %d: %s" % (ep + 1, {k: round(float(v), 4) for k, v in r.items()}))


if __name__ == "__main__":
    main()
