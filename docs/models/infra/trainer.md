# 训练器与统一口径（BaseTrainer）

> **定位**：**唯一**的训练/评估循环 —— 26 个任务共用，口径在这里统一
> **分类**：infra

---

## 一句话定位

`ptcore/trainers/base.py::BaseTrainer` 是**唯一**的训练循环实现。
26 个任务只提供 `TaskAdapter`（数据 / loss / 指标），**循环本身完全共享**。
---

## 为什么这样设计

若每个任务各写一个训练循环，会出现：
- AMP / fp32 处理不一致
- EMA 公式不一致
- 梯度裁剪阈值不一致
- best 指标方向判断不一致

⇒ 本框架把所有**训练语义**收进 `BaseTrainer`，
任务层只描述「这个任务的张量长什么样」。
---

## 核心口径一览

| 口径 | 实现 | 为什么 |
|---|---|---|
| **评估强制 fp32** | `autocast(enabled=False)` | fp16 下 yolo26 框解码 / proto einsum 失真（实测 mask mAP 0.56 vs 0.64） |
| **cudnn.deterministic** | 默认开（`Global.cudnn_deterministic`） | 否则同权重评估抖动，cls 在 conf 阈值翻转 → mAP 低估 |
| **best_metric 方向** | `Global.main_indicator_mode`（min 或 max） | RMSE / MAE / loss 越小越好；原硬编码 ≥ 会判反 |
| **梯度裁剪** | `Optimizer.clip_grad_norm`（默认 10.0，`null` 关闭） | ultra=10；PaddleSpeech 等不裁剪时需关 |
| **EMA** | `use_ema` + `ema_decay_type`（threshold 或 exponential） | ultra 用 exponential（0.9999, tau=2000） |
| **AMP** | `Global.amp`（loss 前 `_to_fp32`，评估强制 fp32） | 加速但不影响数值口径 |
| **梯度累积** | `accumulate = nbs / batch`（读 `Train.loader`） | 有效 batch 决定 LR 相位 |
| **两阶段钩子** | loss 提供 `train_step` 时**完全接管** | USAD / AT 需两次 backward+step |
| **按 batch 评估可关** | `eval_batch_step: null` | 只按 epoch 评估 |
| **loss 必须返回 dict** | trainer 取 `loss_dict` 的 loss 项 | 统一多分量打印 |
---

## 训练循环（伪代码）

```
for epoch:
    for batch in train_loader:
        with autocast(enabled=use_amp):
            preds = task.forward_train(model, images, labels)
        loss_dict = loss(_to_fp32(preds) if use_amp else preds, labels)
        loss = loss_dict[loss 项]

        # ★ 两阶段模型的专用路径
        if hasattr(loss, train_step):
            loss.train_step(model, loss_dict, labels, step_idx)
        else:
            backward(loss)
            if do_step:
                clip_grad_norm_(model.parameters(), clip_grad_norm)
                optimizer.step(); optimizer.zero_grad(); lr_scheduler.step()

        if ema and do_step: ema.update(model)
    if epoch % eval_epoch_step == 0: evaluate_and_save()
```
---

## 评估口径（evaluate_and_save）

1. 临时把 **end2end 头**的 `end2end` 置为 `PostProcess.end2end`（评估后还原）
   —— 使 yolo26 评估走 **one2many + NMS**（与 ultra 一致，实测差 0.002）
2. BN eps 处理：仅 `yolo_cls` 用 1e-5（det / seg / pose / obb 保留 1e-3）
   —— 强改 1e-5 会崩（seg box_mAP50 0.767 → 0.382）
3. 强制 fp32 autocast
4. 累积指标 → `metric.get_metric()`
5. 按 `main_indicator_mode` 判断是否 best → 保存 `best_accuracy.pth`
---

## 产物文件

| 文件 | 内容 |
|---|---|
| `best_accuracy.pth` | 按主指标最优的**模型权重** |
| `latest.pth` / `final.pth` | 最新 / 训练结束 |
| `epoch_N.pth` | 按 `save_epoch_step` |
| `config.yml` | 本次运行的完整配置快照 |
| `train.log` | 训练日志 |
---

## 已知问题

- ⚠️ **TFT / BiLSTM+Attn / TransformerReg 有退出期原生崩溃**（`0xC0000409`）：
  训练与指标正常，孤立前向+反向退出正常 ⇒ 与训练循环后的 CUDA 状态有关，**不影响结果**
- ⚠️ `eval_batch_step: null` 才关闭按 batch 评估
  （写大数如 100000 **无效**，因为 step=0 会命中 `step % interval == 0`）
- ⚠️ **两阶段钩子里不能写 `loss_hist[-1]`**（此时尚未 append）—— 曾导致索引越界崩溃
