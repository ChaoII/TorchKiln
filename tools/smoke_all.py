"""Smoke-test every config end-to-end: model + dataset + 1 train step + 1 eval step.

Usage:
    python tools/smoke_all.py [--pretrained-dir _downloads/official]
"""
import argparse
import glob
import os
import sys
import traceback

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from ptcore.factory import build_trainer
from pytorchx.ocr.utils.config import load_config

UTILS = os.path.join(ROOT, "pytorchx/ocr", "utils")


def char_dict_for(name):
    if "PP-OCRv5" in name:
        return os.path.join(UTILS, "dict", "ppocrv5_dict.txt")
    if "PP-OCRv6_medium" in name or "PP-OCRv6_small" in name:
        return os.path.join(UTILS, "dict", "ppocrv6_dict.txt")
    if "PP-OCRv6_tiny" in name:
        return os.path.join(UTILS, "dict", "ppocrv6_tiny_dict.txt")
    return os.path.join(UTILS, "ppocr_keys_v1.txt")


def one_train_step(trainer):
    loader = trainer.train_loader
    batch = None
    for b in loader:
        if len(b) > 0:
            batch = b
            break
    if batch is None:
        raise RuntimeError("empty train batch")
    images = batch[0].to(trainer.device)
    labels = [x.to(trainer.device) for x in batch]
    preds = trainer._forward_train(images, labels)
    loss_dict = trainer.loss(preds, labels)
    loss = loss_dict["loss"]
    trainer.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 1e9)
    trainer.optimizer.step()
    return float(loss.detach())


def one_eval_step(trainer):
    if trainer.eval_loader is None:
        return None
    trainer.model.eval()
    trainer.metric.reset()
    with torch.no_grad():
        for batch in trainer.eval_loader:
            if len(batch) == 0:
                continue
            trainer.task.eval_step(
                trainer.model, batch, trainer.post_process, trainer.metric, trainer.device
            )
            break
    return trainer.metric.get_metric()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained-dir", default=os.path.join(os.path.expanduser("~"), ".pytorchocr", "pretrained"))
    args = ap.parse_args()

    ok, fail = 0, 0
    for cfg_path in sorted(glob.glob(os.path.join(ROOT, "configs", "*", "*.yml"))):
        name = os.path.splitext(os.path.basename(cfg_path))[0]
        kind = os.path.basename(os.path.dirname(cfg_path))
        if kind == "debug":
            continue
        try:
            config = load_config(cfg_path)
            config["Global"]["epoch_num"] = 1
            config["Global"]["save_model_dir"] = os.path.join(ROOT, "output", "_smoke", name)
            config["Global"]["use_ema"] = False
            config["Global"]["print_batch_step"] = 1
            config.setdefault("Train", {}).setdefault("loader", {})
            config["Train"]["loader"]["batch_size_per_card"] = 4
            config["Train"]["loader"]["num_workers"] = 0
            if config.get("Eval"):
                config["Eval"]["loader"]["num_workers"] = 0
                config["Eval"]["loader"]["batch_size_per_card"] = 2
            pt = os.path.join(args.pretrained_dir, name + "_ptocr.pth")
            config["Global"]["pretrained_model"] = pt if os.path.isfile(pt) else None
            if kind == "rec":
                config["Global"]["character_dict_path"] = char_dict_for(name)

            trainer = build_trainer(config, dump_config=False)
            if trainer.model_type == "rec":
                trainer.model.train()
            loss = one_train_step(trainer)
            metrics = one_eval_step(trainer)
            print("OK   %-28s loss=%.4f eval=%s" % (name, loss, metrics))
            ok += 1
        except Exception as e:
            print("FAIL %-28s %s" % (name, e))
            traceback.print_exc()
            fail += 1
    print("\n{} OK, {} FAIL".format(ok, fail))


if __name__ == "__main__":
    main()
