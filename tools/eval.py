"""评估入口：跑一次评估并把结果写成**机器契约**。

为什么评估也要写 metrics.jsonl
------------------------------
原先这里只 ``logger.info`` 打印两行人读的日志：

    cur metric, box_mAP50: 0.87, mask_mAP50: 0.79, fps: 12.3
    main indicator (hmean): 0.83

于是消费方只能去**正则解析控制台日志**——而这正是 :mod:`ptcore.metrics_sink`
开头列的三种坑（日志格式随版本变、stdout 有缓冲、无 step 对齐）。AIStation 侧
为此写了两个兜底正则（``_parse_torchkiln_metrics``），且还得留一句警告说
「只认标记行的话指标会静默变成空」。

更关键的是：**没有 ``end`` 事件就没法判终态**。TorchKiln 服务侧判定作业成败依赖
``classify_exit(end_event, exit_code, ...)``，而「进程退出且没有 end 事件」会被判成
``no_end_event / failed``。所以评估一旦要作为 HTTP 作业被外部调度，就必须写契约。

所以这里直接用 :class:`ptcore.metrics_sink.MetricSink`，写两条事件：
``eval``（全部指标）与 ``end``（终态 + 主指标）。

另外保留一行 ``EVAL_METRIC_JSON {json}`` 标记行：给人读的日志里能直接看到
结构化结果，也让仍在用日志解析的老消费方不至于立刻断掉。

关闭：环境变量 ``TKILN_METRICS=0`` 或 ``Global.metrics_sink: false``（与训练一致）。
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

from ptcore.factory import build_trainer
from ptcore.metrics_sink import build_sink, sink_enabled
from torchkiln.ocr.utils.config import flatten_opts, parse_args_to_config

#: 供仍在解析日志的消费方使用的一行标记。内容是 ``metrics`` 的 JSON。
#: 前缀选一个几乎不会与训练日志混淆的字符串，便于精确匹配。
METRIC_MARKER = "EVAL_METRIC_JSON "


def _resolve_out_dir(config, trainer):
    """评估产物目录：优先 ``Global.save_model_dir``，回退到 trainer 上的同名属性。

    取不到就返回 None（此时只打日志、不写契约——指标仍会出现在控制台，
    但外部服务会因缺 end 事件而判失败，这是**有意的**：宁可显式失败，
    也不要静默给出一个"成功但没指标"的结果）。
    """
    g = getattr(config, "Global", None) or {}
    out = g.get("save_model_dir") if isinstance(g, dict) else None
    if not out:
        out = getattr(trainer, "save_model_dir", None)
    return out


def main():
    parser = argparse.ArgumentParser(description="TorchKiln evaluation")
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

    # ⚠️ trainer 自己的 sink 在 dump_config=False 下是 _NullSink（见 base.py 的
    #   注释：eval 不拥有训练产物目录）。所以这里**另建**一个写进 eval 的输出目录，
    #   而不是试图打开 trainer.metrics_sink。
    out_dir = _resolve_out_dir(config, trainer)
    sink = build_sink(
        out_dir,
        enabled=sink_enabled((config.get("Global") or {}).get("metrics_sink", True)),
    )

    try:
        metrics = trainer.evaluate()
    except Exception as exc:  # noqa: BLE001
        # 评估本身抛异常也必须收尾写 end，否则外部服务会一直等下去。
        sink.end(exit_reason="error", error="{}: {}".format(type(exc).__name__, exc))
        raise

    if metrics is None:
        trainer.logger.warning("No Eval dataset configured; nothing to evaluate.")
        sink.end(exit_reason="error",
                 error="未配置 Eval 数据集，无可评估内容（检查 Eval.dataset.data_dir "
                       "与 label_file_list）")
        # 非零退出：让调用方一眼看出"没跑起来"而不是"跑了但指标为空"
        sys.exit(2)

    fps = metrics.get("fps", 0.0)
    main_key = (config.get("Metric") or {}).get("main_indicator", "hmean")
    main_value = float(metrics.get(main_key, 0.0))

    trainer.logger.info(
        "cur metric, %s, fps: %s",
        ", ".join(
            "{}: {}".format(k, float(v)) for k, v in metrics.items() if k != "fps"
        ),
        fps,
    )
    trainer.logger.info(
        "main indicator (%s): %s", main_key, main_value
    )

    # 结构化标记行：给人看，也给仍在解析日志的消费方留个过渡期。
    # 契约在 MetricsSink 文档里；这里保证与 jsonl 里的 eval 事件同源。
    payload = dict(metrics)
    payload.setdefault("fps", fps)
    payload["main_indicator"] = main_key
    payload["main_value"] = main_value
    trainer.logger.info(METRIC_MARKER + json.dumps(payload, ensure_ascii=False,
                                                    allow_nan=False))

    # 机器契约：eval 一条（全指标）+ end 一条（终态）。
    # ⚠️ 顺序不能反：end 会 close 文件。
    sink.eval(main_indicator=main_key, main_value=main_value, fps=fps,
              metrics={k: v for k, v in metrics.items() if k != "fps"})
    sink.end(exit_reason="finished", main_indicator=main_key, main_value=main_value)


if __name__ == "__main__":
    main()