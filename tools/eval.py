from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

from ptcore.factory import build_trainer
from pytorchx.ocr.utils.config import flatten_opts, parse_args_to_config


def main():
    parser = argparse.ArgumentParser(description="PyTorchOCR evaluation")
    parser.add_argument("-c", "--config", required=True, help="config yaml path")
    parser.add_argument(
        "-o", "--opt", nargs="*", action="append", default=None,
        help="config overrides, repeatable: -o a=1 -o b=2",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="checkpoint to evaluate (overrides Global.pretrained_model)",
    )
    args = parser.parse_args()

    config = parse_args_to_config(args.config, flatten_opts(args.opt))
    if args.weights:
        config["Global"]["pretrained_model"] = args.weights
    config["Global"]["epoch_num"] = 0
    config["Global"]["use_ema"] = False
    config["Global"].setdefault("save_model_dir", "./output/_eval")

    trainer = build_trainer(config, dump_config=False)
    trainer._log_summary()
    metrics = trainer.evaluate()
    if metrics is None:
        trainer.logger.warning("No Eval dataset configured; nothing to evaluate.")
        return
    fps = metrics.get("fps", 0.0)
    main = config.get("Metric", {}).get("main_indicator", "hmean")
    trainer.logger.info(
        "cur metric, %s, fps: %s",
        ", ".join(
            "{}: {}".format(k, float(v)) for k, v in metrics.items() if k != "fps"
        ),
        fps,
    )
    trainer.logger.info(
        "main indicator (%s): %s", main, float(metrics.get(main, 0.0))
    )


if __name__ == "__main__":
    main()
