from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import datetime
import logging
import os
import platform
import time
from collections import deque

import numpy as np
import torch
from torch.utils.data import DataLoader

from ptcore.batch_sampler import PaddleBatchSampler
from ptcore.ema import ModelEMA
from ptcore.optimizer import build_optimizer
from ptcore.precision import enable_paddle_like_precision
from ptcore.task import TaskAdapter


def _to_fp32(x):
    """Recursively cast floating tensors (model outputs) to float32."""
    if isinstance(x, torch.Tensor):
        return x.float() if x.is_floating_point() else x
    if isinstance(x, dict):
        return {k: _to_fp32(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_to_fp32(v) for v in x)
    return x


def get_logger(name, log_file=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y/%m/%d %H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


class BaseTrainer:
    """Task-agnostic trainer.

    Everything task specific is delegated to a :class:`~ptcore.task.TaskAdapter`
    (``self.task``). Subclasses only need to provide ``_default_task``.
    """

    def __init__(self, config, dump_config=True, task=None):
        self.config = config
        self.global_config = config["Global"]
        gcfg = self.global_config

        enable_paddle_like_precision()

        # cuDNN 非确定性算法会让我们的 GPU 前向与 ultralytics 产生不同数值
        # (cls 在 conf 阈值附近大量翻转),导致评估 mAP 被系统性低估。
        # 强制 deterministic 使 GPU 结果与 CPU/ultralytics 逐点一致(默认开)。
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = bool(gcfg.get("cudnn_deterministic", True))

        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.distributed = self.world_size > 1
        self.is_main = self.rank == 0

        self.task = task if task is not None else self._default_task(config)
        self.task_name = getattr(self.task, "name", "task")

        if gcfg.get("output"):
            gcfg["save_model_dir"] = gcfg["output"]
        self.save_model_dir = gcfg.get("save_model_dir", "./output")
        if self.is_main:
            os.makedirs(self.save_model_dir, exist_ok=True)
        self.logger = get_logger(
            "pytorchx/ocr",
            os.path.join(self.save_model_dir, "train.log") if self.is_main else None,
        )
        if self.distributed:
            self.logger.info(
                "Distributed run: rank=%d local_rank=%d world_size=%d",
                self.rank,
                self.local_rank,
                self.world_size,
            )

        dev = gcfg.get("device")
        self.device_ids = []
        explicit_ids = []
        dev_str = str(dev) if dev is not None else ""
        if dev_str.startswith("gpu"):
            id_str = dev_str.split(":")[1] if ":" in dev_str else "0"
            explicit_ids = [int(x) for x in id_str.split(",") if x != ""]
        elif dev_str.startswith("cuda"):
            explicit_ids = [int(dev_str.split(":")[1])] if ":" in dev_str else [0]

        if self.distributed:
            if explicit_ids:
                self.device_ids = explicit_ids
                idx = explicit_ids[min(self.local_rank, len(explicit_ids) - 1)]
                self.device = torch.device("cuda:{}".format(idx))
            elif dev_str.startswith("cpu") or not torch.cuda.is_available():
                self.device = torch.device("cpu")
                self.device_ids = []
            else:
                idx = self.local_rank
                self.device_ids = [idx]
                self.device = torch.device("cuda:{}".format(idx))
        elif explicit_ids:
            self.device_ids = explicit_ids
            self.device = torch.device("cuda:{}".format(explicit_ids[0]))
        elif dev_str and not dev_str.startswith(("gpu", "cuda")):
            self.device = torch.device("cpu")
        else:
            use_gpu = gcfg.get("use_gpu", True)
            self.device = (
                torch.device("cuda:0")
                if use_gpu and torch.cuda.is_available()
                else torch.device("cpu")
            )
            if self.device.type == "cuda":
                self.device_ids = [0]
        if self.device.type == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")
            self.device_ids = []

        if self.distributed:
            backend = gcfg.get("dist_backend")
            if not backend:
                backend = "nccl" if torch.distributed.is_nccl_available() else "gloo"
            if backend == "gloo" and self.device.type == "cuda":
                self.logger.warning(
                    "NCCL unavailable; falling back to gloo on CUDA (slow). "
                    "Use an NCCL-capable build for efficient multi-GPU DDP."
                )
            init_method = gcfg.get("dist_init_method") or "env://"
            torch.distributed.init_process_group(
                backend=backend,
                init_method=init_method,
                rank=self.rank,
                world_size=self.world_size,
            )
            if self.device.type == "cuda":
                torch.cuda.set_device(self.device)

        if dump_config:
            self._dump_config(config)

        self.use_data_parallel = False

        self.model_type = self.task_name
        self.seed = int(gcfg.get("seed", 1024))
        self._set_seed(self.seed)

        self.overrides = list(config.get("_overrides", []) or [])
        self.epoch_num = int(gcfg.get("epoch_num", 100))
        self.checkpoint_path = None
        self._checkpoint_state = None
        resume = gcfg.get("checkpoints") or config.get("Train", {}).get("resume_path")
        if resume:
            rpath = resume[-1] if isinstance(resume, (list, tuple)) else resume
            if isinstance(rpath, str) and os.path.isfile(rpath):
                self.checkpoint_path = rpath
                self._checkpoint_state = torch.load(rpath, map_location="cpu")
                saved_epochs = (
                    self._checkpoint_state.get("epoch_num")
                    if isinstance(self._checkpoint_state, dict)
                    else None
                )
                if saved_epochs and "Global.epoch_num" not in self.overrides:
                    self.epoch_num = int(saved_epochs)
                    self.logger.info(
                        "Restored epoch_num=%d from %s "
                        "(pass -o Global.epoch_num=... to override).",
                        self.epoch_num,
                        os.path.basename(rpath),
                    )
        self.print_batch_step = int(gcfg.get("print_batch_step", 10))
        self.eval_batch_step = gcfg.get("eval_batch_step", [0, 1500])
        self.save_epoch_step = int(gcfg.get("save_epoch_step", 10))
        self.eval_epoch_step = max(1, int(gcfg.get("eval_epoch_step", 1)))
        self.use_ema = bool(gcfg.get("use_ema", False))
        self.ema_decay = float(gcfg.get("ema_decay", 0.9998))
        self.ema_decay_type = gcfg.get("ema_decay_type", "threshold")

        self.logger.info("Building postprocess...")
        self.post_process = self.task.build_post_process(config)

        self.logger.info("Building model...")
        self.model = self.task.build_model(config, self.post_process).to(self.device)
        self._sync_head_strides(config)
        self._load_pretrained(gcfg.get("pretrained_model"))
        self._wrap_model()
        self._apply_freeze(gcfg.get("freeze"))

        self.logger.info("Building loss / metric...")
        self.loss = self.task.build_loss(config, self._raw_model()).to(self.device)
        self.metric = self.task.build_metric(config)

        self.logger.info("Building dataset / dataloader...")
        collate = self.task.train_collate
        eval_collate = self.task.eval_collate
        train_ds_cfg = config["Train"]["dataset"]
        self.logger.info(
            "Train dataset: data_dir=%s label_file_list=%s",
            train_ds_cfg.get("data_dir"),
            train_ds_cfg.get("label_file_list"),
        )
        if config.get("Eval") is not None:
            eval_ds_cfg = config["Eval"]["dataset"]
            self.logger.info(
                "Eval dataset: data_dir=%s label_file_list=%s",
                eval_ds_cfg.get("data_dir"),
                eval_ds_cfg.get("label_file_list"),
            )
        self.train_dataset, self.eval_dataset = self.task.build_datasets(
            config, self.logger
        )
        self.train_batch_sampler = PaddleBatchSampler(
            self.train_dataset,
            batch_size=int(config["Train"]["loader"]["batch_size_per_card"]),
            shuffle=bool(config["Train"]["loader"]["shuffle"]),
            drop_last=bool(config["Train"]["loader"].get("drop_last", False)),
            num_replicas=self.world_size,
            rank=self.rank,
        )
        train_num_workers = int(config["Train"]["loader"]["num_workers"])
        train_loader_kwargs = {}
        if train_num_workers > 0:
            # Spawning workers is expensive (especially on Windows); keep them
            # alive across epochs and let them prefetch ahead.
            train_loader_kwargs["persistent_workers"] = True
            train_loader_kwargs["prefetch_factor"] = int(
                config["Train"]["loader"].get("prefetch_factor", 4)
            )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_sampler=self.train_batch_sampler,
            num_workers=train_num_workers,
            collate_fn=collate,
            pin_memory=True,
            **train_loader_kwargs,
        )
        self.eval_loader = None
        if self.eval_dataset is not None:
            eval_num_workers = int(config["Eval"]["loader"]["num_workers"])
            eval_loader_kwargs = {}
            if eval_num_workers > 0:
                eval_loader_kwargs["persistent_workers"] = True
                eval_loader_kwargs["prefetch_factor"] = int(
                    config["Eval"]["loader"].get("prefetch_factor", 4)
                )
            self.eval_loader = DataLoader(
                self.eval_dataset,
                batch_size=int(config["Eval"]["loader"]["batch_size_per_card"]),
                shuffle=False,
                num_workers=eval_num_workers,
                collate_fn=eval_collate,
                drop_last=False,
                **eval_loader_kwargs,
            )

        self.steps_per_epoch = max(1, len(self.train_loader))
        # 对齐 ultralytics:梯度累积由 nbs/batch 推导(Global.accumulate 可覆盖)
        _bs = int(
            (((config.get("Train") or {}).get("dataset") or {}).get("loader") or {}).get(
                "batch_size_per_card", 1
            )
            or 1
        )
        _nbs = (config.get("Optimizer") or {}).get("nbs")
        if gcfg.get("accumulate") is not None:
            self.accumulate = max(1, int(gcfg.get("accumulate")))
        elif _nbs:
            self.accumulate = max(1, int(round(float(_nbs) / max(_bs, 1))))
        else:
            self.accumulate = 1
        # LR 调度按 **optimizer 步** 推进(= epoch 进度),否则 accumulate>1 时步数骤减、
        # 调度不会衰减,学习率长期停在峰值 -> 收敛远慢于原版。
        # 注意 Windows 下训练循环只取 len(loader)-1 个 batch(PaddleOCR 语义),需据此折算。
        _win = platform.system() == "Windows"
        _bpe = max(1, len(self.train_loader) - 1 if _win else len(self.train_loader))
        # do_step 在 ``(idx+1)%accumulate==0 or idx+1>=max_iter`` 时触发,故每 epoch 实为
        # ``ceil(max_iter/accumulate)`` 次(向下取整会低估,导致 LR 相位错乱)。
        self.opt_steps_per_epoch = max(1, (_bpe + self.accumulate - 1) // self.accumulate)
        self.logger.info("Building optimizer...")
        self.optimizer, self.lr_scheduler = build_optimizer(
            config["Optimizer"], self.epoch_num, self.opt_steps_per_epoch, self._raw_model()
        )

        self.ema = None
        if self.use_ema:
            self.ema = ModelEMA(
                self._raw_model(),
                decay=self.ema_decay,
                ema_decay_type=self.ema_decay_type,
            )
        self.ema_model = None

        # mixed precision (opt-in; keeps fp32 master weights)
        self.use_amp = bool(gcfg.get("amp", False)) and self.device.type == "cuda"
        self.scaler = None
        if self.use_amp:
            self.scaler = torch.amp.GradScaler("cuda", enabled=True)
        self.amp_dtype = (
            torch.float16
            if str(gcfg.get("amp_dtype", "fp16")).lower() in ("fp16", "half")
            else torch.bfloat16
        )

        self.global_step = 0
        self._last_eval_global_step = -1
        self.best_metric = 0.0
        self.best_metrics = {}
        self.best_fps = 0.0
        self.best_epoch = 0
        self.is_float16 = False
        self.show_progress = bool(gcfg.get("show_eval_progress", True))
        self.current_epoch = 0
        self.start_epoch = 0
        self.resumed = False
        self.log_smooth_window = max(1, int(gcfg.get("log_smooth_window", 20)))
        self.main_indicator = config.get("Metric", {}).get("main_indicator", "hmean")
        self._load_resume()

    def _load_resume(self):
        state = self._checkpoint_state
        path = self.checkpoint_path
        if state is None or path is None:
            return
        if isinstance(state, dict) and "model" in state:
            self._raw_model().load_state_dict(state["model"], strict=False)
            if "optimizer" in state:
                try:
                    self.optimizer.load_state_dict(state["optimizer"])
                except Exception:
                    self.logger.warning("Failed to restore optimizer state.")
            if "lr_scheduler" in state:
                try:
                    self.lr_scheduler.load_state_dict(state["lr_scheduler"])
                except Exception:
                    self.logger.warning("Failed to restore lr_scheduler state.")
            if "ema" in state and self.ema is not None:
                self.ema.set_state_dict(state["ema"])
            self.global_step = int(state.get("global_step", 0))
            self.best_metric = float(state.get("best_metric", 0.0))
            self.best_metrics = state.get("best_metrics", {}) or {}
            self.best_fps = float(state.get("best_fps", 0.0))
            self.best_epoch = int(state.get("best_epoch", 0))
            # `epoch` may be the legacy sentinel -1 from older checkpoints.
            self.start_epoch = max(0, int(state.get("epoch", 0) or 0))
            self.best_epoch = max(0, self.best_epoch)
            self.resumed = True
            self.logger.info(
                "Resumed from %s (epoch=%d, global_step=%d, best_%s=%.5f @epoch %d)",
                path,
                self.start_epoch,
                self.global_step,
                self.main_indicator,
                self.best_metric,
                self.best_epoch,
            )
        else:
            self._raw_model().load_state_dict(state, strict=False)
            self.logger.info("Resumed model weights from %s", path)

    def _default_task(self, config):
        raise NotImplementedError(
            "BaseTrainer needs a task adapter; pass task=... or subclass and "
            "implement _default_task()."
        )

    def _dump_config(self, config):
        if not self.is_main:
            return
        try:
            import yaml

            path = os.path.join(self.save_model_dir, "config.yml")
            dump = copy.deepcopy(config)
            overrides = dump.pop("_overrides", None)
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(dump, f, sort_keys=False, allow_unicode=True)
            if overrides:
                self.logger.info("Overrides applied: %s", ", ".join(overrides))
            self.logger.info("Config saved to %s", path)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Failed to save config: %s", exc)

    def _log_summary(self):
        g = self.global_config
        arch = self.config.get("Architecture", {})
        opt = self.config.get("Optimizer", {})
        lr_cfg = opt.get("lr", {}) if isinstance(opt.get("lr"), dict) else {}
        tr = self.config.get("Train", {}).get("loader", {})
        ev = (self.config.get("Eval") or {}).get("loader", {})
        head = arch.get("Head") or {}
        self.logger.info(
            "Config: model=%s type=%s algorithm=%s",
            g.get("model_name"),
            arch.get("model_type"),
            arch.get("algorithm"),
        )
        self.logger.info(
            "Config: epochs=%s steps/epoch=%s batch=%s eval_batch=%s workers=%s/%s "
            "seed=%s device=%s use_ema=%s distributed=%s amp=%s",
            self.epoch_num,
            self.steps_per_epoch,
            tr.get("batch_size_per_card"),
            ev.get("batch_size_per_card"),
            tr.get("num_workers"),
            ev.get("num_workers"),
            self.seed,
            self.device,
            self.use_ema,
            self.distributed,
            self.use_amp,
        )
        self.logger.info(
            "Config: optimizer=%s lr=%s warmup_epoch=%s l2=%s",
            opt.get("name"),
            lr_cfg.get("learning_rate"),
            lr_cfg.get("warmup_epoch"),
            (opt.get("regularizer") or {}).get("factor"),
        )
        if head.get("name"):
            self.logger.info(
                "Config: head=%s backbone=%s neck=%s",
                head.get("name"),
                (arch.get("Backbone") or {}).get("name"),
                (arch.get("Neck") or {}).get("name"),
            )
        for line in self.task.summary_lines(self.config, g, self.post_process) or []:
            if line:
                self.logger.info("Config: %s", line)
        self.logger.info(
            "Config: pretrained=%s checkpoints=%s save_dir=%s",
            g.get("pretrained_model"),
            g.get("checkpoints"),
            self.save_model_dir,
        )

    def _set_seed(self, seed):
        import random

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 可复现模式(默认关闭,避免影响其它模型的训练速度):
        # 用于 A/B 对照,消除 cudnn 非确定性带来的 mAP 抖动。
        if bool((self.global_config or {}).get("deterministic", False)):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _sync_head_strides(self, config):
        """Align anchor strides with the parsed head levels.

        Multi-level variants (``*-p2`` -> P2-P5, ``*-p6`` -> P3-P6) do not match
        the ``[8, 16, 32]`` default, which would misalign the anchor grid used by
        the loss / post-process / metric.
        """
        model = self.model
        head = getattr(model, "head", None)
        stride = getattr(model, "stride", None)
        if head is None or stride is None or not hasattr(stride, "flatten"):
            return
        vals = [int(round(float(v))) for v in stride.flatten().tolist()]
        vals = [v for v in vals if v > 0]
        nl = getattr(head, "nl", None)
        if not vals or (nl is not None and len(vals) != int(nl)):
            return
        for section in ("PostProcess", "Metric", "Loss"):
            sec = config.get(section)
            if isinstance(sec, dict):
                sec["strides"] = list(vals)
        if getattr(self.post_process, "strides", None) is not None:
            self.post_process.strides = tuple(vals)
        self.logger.info("Head strides: %s", vals)

    def _load_pretrained(self, pretrained):
        if not pretrained:
            self.logger.info(
                "No pretrained weights specified (training from scratch)."
            )
            return
        from ptcore.pretrained import resolve_pretrained

        try:
            path = resolve_pretrained(pretrained, logger=self.logger)
        except FileNotFoundError as exc:
            self.logger.error("%s", exc)
            raise
        if not path:
            self.logger.warning(
                "Pretrained weights '%s' not found and could not be downloaded "
                "(training from scratch).",
                pretrained,
            )
            return
        self.logger.info("Pretrained source: %s -> %s", pretrained, path)
        from ptcore.pretrained import load_state_dict_any

        # 支持:纯张量 dict / {'state_dict':...} / 训练断点 / 上游 ultralytics .pt
        state = load_state_dict_any(path)
        if isinstance(state, dict):
            if "state_dict" in state:
                state = state["state_dict"]
            elif isinstance(state.get("model"), dict):
                # our own `latest.pth` (model + optimizer + ema + ...)
                state = state["model"]

        # Like PaddleOCR's load_pretrained_params(): skip weights whose shape
        # does not match (typically the classification head when the character
        # dict differs) instead of aborting the whole run.
        own = self.model.state_dict()
        filtered = {}
        skipped_shape = []
        for key, value in state.items():
            if key not in own:
                continue
            if tuple(own[key].shape) != tuple(value.shape):
                skipped_shape.append((key, tuple(value.shape), tuple(own[key].shape)))
                continue
            filtered[key] = value
        if skipped_shape:
            self.logger.warning(
                "%d pretrained param(s) skipped because of shape mismatch "
                "(e.g. %s %s -> %s).",
                len(skipped_shape),
                skipped_shape[0][0],
                skipped_shape[0][1],
                skipped_shape[0][2],
            )
            head_like = [s for s in skipped_shape if "head" in s[0]]
            if head_like:
                self.logger.warning(
                    "The classification head differs, which means num_classes "
                    "(= character_dict_path) is not the same as the pretrained "
                    "model's. Two cases: (a) fine-tuning on a NEW charset "
                    "(e.g. digits only) -> expected, the head is re-initialised "
                    "and trained from scratch; (b) you meant to fine-tune the "
                    "official charset -> set the matching dict, e.g. "
                    "pytorchx/ocr/utils/dict/ppocrv6_tiny_dict.txt for "
                    "PP-OCRv6_tiny_rec."
                )
        missing, unexpected = self.model.load_state_dict(filtered, strict=False)
        note = ""
        if missing:
            benign = [
                k
                for k in missing
                if k.endswith("num_batches_tracked")
                or "aux_binarize" in k
                or "aux_thresh" in k
            ]
            if len(benign) == len(missing):
                note = " (all training-only: BN counters / aux branches; inference unaffected)"
        self.logger.info(
            "Loaded pretrained: %s (missing=%d unexpected=%d)%s",
            path,
            len(missing),
            len(unexpected),
            note,
        )
        if missing and not note:
            self.logger.warning(
                "Some weights were NOT loaded (kept random): %s",
                ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else ""),
            )
        if unexpected:
            self.logger.warning(
                "Checkpoint has unexpected weights (ignored): %s",
                ", ".join(unexpected[:10]) + (" ..." if len(unexpected) > 10 else ""),
            )

    def _apply_freeze(self, freeze):
        """Freeze parameters: ``freeze: N`` (first N graph layers) or name prefixes."""
        if not freeze:
            return
        total = frozen = 0
        for name, p in self._raw_model().named_parameters():
            total += 1
            hit = False
            if isinstance(freeze, (int, float)) and not isinstance(freeze, bool):
                parts = name.split(".")
                if len(parts) > 1:
                    try:
                        hit = int(parts[1]) < int(freeze)
                    except ValueError:
                        hit = False
            else:
                prefixes = freeze if isinstance(freeze, (list, tuple)) else [freeze]
                hit = any(name.startswith(str(pre)) for pre in prefixes)
            if hit:
                p.requires_grad = False
                frozen += 1
        self.logger.info("Freeze: %d/%d params have requires_grad=False", frozen, total)

    def _raw_model(self):
        model = self.model
        while isinstance(
            model,
            (
                torch.nn.parallel.DistributedDataParallel,
                torch.nn.DataParallel,
            ),
        ):
            model = model.module
        return model

    def _wrap_model(self):
        if self.distributed:
            ddp_kwargs = {"find_unused_parameters": True}
            if self.device.type == "cuda":
                ddp_kwargs["device_ids"] = [self.device.index]
                ddp_kwargs["output_device"] = self.device.index
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, **ddp_kwargs
            )
            self.logger.info("Using DistributedDataParallel (DDP).")
        elif len(self.device_ids) > 1:
            self.use_data_parallel = True
            self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)
            self.logger.info(
                "Using DataParallel on GPUs %s (single-process). For better "
                "scaling launch with: torchrun --nproc_per_node=%d tools/train.py ...",
                self.device_ids,
                len(self.device_ids),
            )

    def _forward_train(self, images, batch):
        return self.task.forward_train(self.model, images, batch)

    def _mem_stats(self):
        if self.device.type != "cuda":
            return 0, 0
        return (
            int(torch.cuda.max_memory_reserved() / (1024 * 1024)),
            int(torch.cuda.max_memory_allocated() / (1024 * 1024)),
        )

    def _train_one_epoch(self, epoch):
        self.model.train()
        tic = time.time()
        window = self.log_smooth_window
        loss_hist = deque(maxlen=window)
        reader_hist = deque(maxlen=window)
        batch_hist = deque(maxlen=window)
        sample_hist = deque(maxlen=window)
        comp_hists = {}
        # PaddleOCR on Windows processes len(dataloader)-1 batches per epoch.
        windows = platform.system() == "Windows"
        max_iter = len(self.train_loader) - 1 if windows else len(self.train_loader)
        last_end = time.time()
        for idx, batch in enumerate(self.train_loader):
            if idx >= max_iter:
                break
            reader_cost = time.time() - last_end
            if len(batch) == 0:
                last_end = time.time()
                continue
            images = batch[0].to(self.device, non_blocking=True)
            labels = [b.to(self.device, non_blocking=True) for b in batch]

            batch_tic = time.time()
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                preds = self._forward_train(images, labels)
            # 损失在 fp32 下计算:自定义算子(CIoU/DFL 等)在 fp16 autocast 下会溢出,
            # 导致 AMP 训练退化(实测 mosaic+累積时 mAP=0)。
            loss_dict = self.loss(_to_fp32(preds) if self.use_amp else preds, labels)
            loss = loss_dict["loss"]

            micro = self.accumulate
            do_step = ((idx + 1) % micro == 0) or (idx + 1 >= max_iter)
            # 对齐 ultralytics:梯度**累积求和**(不除以 accumulate);每步做梯度裁剪(unscale+clip=10)。
            scaled = loss
            if self.scaler is not None:
                self.scaler.scale(scaled).backward()
                if do_step:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self._raw_model().parameters(), max_norm=10.0, foreach=False
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.lr_scheduler.step()
            else:
                scaled.backward()
                if do_step:
                    torch.nn.utils.clip_grad_norm_(
                        self._raw_model().parameters(), max_norm=10.0, foreach=False
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.lr_scheduler.step()
            if self.ema is not None and do_step:
                self.ema.update(self._raw_model())
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            batch_cost = time.time() - batch_tic
            last_end = time.time()

            if do_step:
                self.global_step += 1
            num_samples = int(images.shape[0])
            loss_hist.append(float(loss.detach()))
            reader_hist.append(reader_cost)
            batch_hist.append(batch_cost)
            sample_hist.append(num_samples)
            for k, v in loss_dict.items():
                if k == "loss":
                    continue
                comp_hists.setdefault(k, deque(maxlen=window)).append(float(v.detach()))

            if idx == 0:
                self.logger.info(
                    "DEBUG batch0 label sums: %s comps: %s",
                    [round(float(b.sum()), 1) for b in labels],
                    {k: round(float(v), 4) for k, v in loss_dict.items()},
                )

            if self.global_step % self.print_batch_step == 0:
                avg_reader = sum(reader_hist) / len(reader_hist)
                avg_batch = sum(batch_hist) / len(batch_hist)
                avg_samples = sum(sample_hist) / len(sample_hist)
                avg_loss = sum(loss_hist) / len(loss_hist)
                lr = float(self.optimizer.param_groups[0]["lr"])
                ips = sum(sample_hist) / max(1e-6, sum(batch_hist))
                remain = max(0, self.epoch_num * self.opt_steps_per_epoch - self.global_step)
                eta = str(datetime.timedelta(seconds=int(remain * avg_batch)))
                mem_res, mem_alloc = self._mem_stats()
                comps = "".join(
                    ", {}: {:.6f}".format(k, sum(h) / len(h))
                    for k, h in comp_hists.items()
                )
                self.logger.info(
                    "epoch: [%d/%d], global_step: %d, lr: %.6f, loss: %.6f%s, "
                    "avg_reader_cost: %.5f s, avg_batch_cost: %.5f s, "
                    "avg_samples: %.5f, ips: %.5f samples/s, eta: %s, "
                    "max_mem_reserved: %d MB, max_mem_allocated: %d MB",
                    epoch + 1,
                    self.epoch_num,
                    self.global_step,
                    lr,
                    avg_loss,
                    comps,
                    avg_reader,
                    avg_batch,
                    avg_samples,
                    ips,
                    eta,
                    mem_res,
                    mem_alloc,
                )
            self._maybe_eval_and_save()
        self.logger.info(
            "epoch: [%d/%d], done, time: %.1fs",
            epoch + 1,
            self.epoch_num,
            time.time() - tic,
        )

    def _maybe_eval_and_save(self):
        evs = self.eval_batch_step
        step = self.global_step
        interval = int(evs[1]) if len(evs) > 1 else 1500
        start = int(evs[0]) if len(evs) > 0 else 0
        if interval <= 0 or step < start or step % interval != 0:
            return
        # 梯度累积窗口内 global_step 会连续多批停在同一个值,
        # 直接 `step % interval == 0` 会在 step=start(通常为0)时每批都触发评估。
        # 只在 global_step 变化到新的满足条件的值时评估一次(否则累积窗口内会重复评估)。
        if step == self._last_eval_global_step:
            return
        self._last_eval_global_step = step
        self.evaluate_and_save()

    def evaluate(self):
        if self.eval_loader is None:
            return None
        raw = self._raw_model()
        model = raw
        if self.ema is not None:
            if self.ema_model is None:
                self.ema_model = copy.deepcopy(raw)
            self.ema_model.load_state_dict(self.ema.apply())
            model = self.ema_model
        model.eval()
        # 对齐 ultralytics 推理数值：ultra 训练 initialize_weights 把 BN eps 设为 1e-3，
        # 但保存的权重不含 BN eps，加载后推理时 BN 用构造默认 1e-5。此处评估临时恢复 1e-5，
        # 评估完再还原（raw 训练模型保持 1e-3，不影响训练 forward）。
        import torch.nn as _nn

        _bns = [m for m in model.modules() if isinstance(m, _nn.BatchNorm2d)]
        _old_bn = [(m, m.eps) for m in _bns]
        for _m in _bns:
            _m.eps = 1e-5
        try:
            return self._evaluate_loop(model, tic=time.time())
        finally:
            for _m, _eps in _old_bn:
                _m.eps = _eps

    def _evaluate_loop(self, model, tic):
        self.metric.reset()
        tic = tic if tic is not None else time.time()
        total_samples = 0
        bar = None
        if self.show_progress:
            try:
                from tqdm import tqdm

                total = len(self.eval_dataset) if self.eval_dataset is not None else None
                bar = tqdm(
                    total=total,
                    desc="eval model::",
                    unit="img",
                    disable=not self.is_main,
                    leave=True,
                )
            except Exception:
                bar = None
        with torch.no_grad():
            for batch in self.eval_loader:
                if len(batch) == 0:
                    continue
                n = self.task.sample_count(batch)
                total_samples += n
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.use_amp,
                ):
                    self.task.eval_step(
                        model, batch, self.post_process, self.metric, self.device
                    )
                if bar is not None:
                    bar.update(n)
        if bar is not None:
            bar.close()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = max(1e-6, time.time() - tic)
        metrics = self.metric.get_metric()
        metrics["fps"] = total_samples / elapsed
        return metrics

    def _is_float16(self):
        try:
            return next(self._raw_model().parameters()).dtype == torch.float16
        except StopIteration:
            return False

    def evaluate_and_save(self):
        if not self.is_main:
            return
        metrics = self.evaluate()
        if metrics is None:
            return
        self.is_float16 = self._is_float16()
        fps = metrics.pop("fps", 0.0)
        self.logger.info(
            "cur metric, %s, fps: %s",
            ", ".join("{}: {}".format(k, float(v)) for k, v in metrics.items()),
            fps,
        )
        value = metrics.get(self.main_indicator, metrics.get("hmean", 0.0))
        if value >= self.best_metric:
            self.best_metric = value
            self.best_epoch = self.current_epoch
            self.best_metrics = dict(metrics)
            self.best_fps = fps
            self.save_checkpoint("best_accuracy", is_best=True)
        best = self.best_metrics or dict(metrics)
        others = ", ".join(
            "{}: {}".format(k, float(v))
            for k, v in best.items()
            if k != self.main_indicator
        )
        self.logger.info(
            "best metric, %s: %s, is_float16: %s, %s, fps: %s, best_epoch: %d",
            self.main_indicator,
            best.get(self.main_indicator, value),
            self.is_float16,
            others,
            self.best_fps if self.best_metrics else fps,
            self.best_epoch if self.best_metrics else self.current_epoch,
        )

    def save_checkpoint(self, name, is_best=False):
        if not self.is_main:
            return
        raw = self._raw_model()
        if is_best:
            state = raw.state_dict()
            if self.ema is not None:
                state = self.ema.apply()
        else:
            state = raw.state_dict()
        path = os.path.join(self.save_model_dir, "{}.pth".format(name))
        torch.save(state, path)
        latest = {
            "model": raw.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "best_metrics": self.best_metrics,
            "best_fps": self.best_fps,
            "best_epoch": self.best_epoch,
            "epoch": self.current_epoch,
            "epoch_num": self.epoch_num,
        }
        if self.ema is not None:
            latest["ema"] = self.ema.state_dict_for_save()
        torch.save(latest, os.path.join(self.save_model_dir, "latest.pth"))

    def train(self):
        self.logger.info(
            "Start training: type=%s epochs=%d steps/epoch=%d device=%s",
            self.model_type,
            self.epoch_num,
            self.steps_per_epoch,
            self.device,
        )
        self._log_summary()
        if self.resumed and self.start_epoch >= self.epoch_num:
            self.logger.warning(
                "Checkpoint already trained %d/%d epochs; nothing to do. "
                "Pass -o Global.epoch_num=<larger> to train more.",
                self.start_epoch,
                self.epoch_num,
            )
            return
        if self.resumed:
            self.train_batch_sampler.set_epoch(self.start_epoch)
            self.logger.info(
                "Continue training from epoch %d to %d.",
                self.start_epoch + 1,
                self.epoch_num,
            )
        patience = int((self.config.get("Global") or {}).get("patience", 0) or 0)
        _last_best, _no_improve = self.best_metric, 0
        for epoch in range(self.start_epoch, self.epoch_num):
            self.current_epoch = epoch + 1
            if hasattr(self.train_dataset, "set_epoch"):
                # e.g. close_mosaic for YOLO-style augmentation schedules
                self.train_dataset.set_epoch(self.current_epoch)
            self._train_one_epoch(epoch)
            if (epoch + 1) % self.save_epoch_step == 0:
                self.save_checkpoint("epoch_{}".format(epoch + 1))
                self.logger.info("saved epoch_%d checkpoint", epoch + 1)
            if self.eval_loader is not None and (epoch + 1) % self.eval_epoch_step == 0:
                self.evaluate_and_save()
            # ultralytics 风格早停:连续 patience 个 epoch 主指标无提升则停
            if self.best_metric > _last_best + 1e-12:
                _last_best, _no_improve = self.best_metric, 0
            else:
                _no_improve += 1
                if patience and _no_improve >= patience:
                    self.logger.info(
                        "EarlyStopping: no improvement for %d epoch(s) (best %s=%.5f), stop at epoch %d",
                        patience, self.main_indicator, self.best_metric, epoch + 1)
                    break
        self.save_checkpoint("final")
        if self.eval_loader is not None:
            self.evaluate_and_save()
        self.logger.info(
            "Training finished. Best %s = %.5f", self.main_indicator, self.best_metric
        )
        if self.distributed:
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
