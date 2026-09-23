from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

from ptcore.factory import build_trainer
from torchkiln.ocr.utils.config import flatten_opts, parse_args_to_config


def main():
    parser = argparse.ArgumentParser(description="TorchKiln training")
    parser.add_argument("-c", "--config", required=True, help="config yaml path")
    parser.add_argument(
        "-o",
        "--opt",
        nargs="*",
        action="append",
        default=None,
        help="override config, repeatable: -o a=1 -o b=2 (also -o a=1 b=2)",
    )
    args = parser.parse_args()
    config = parse_args_to_config(args.config, flatten_opts(args.opt))
    trainer = build_trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
