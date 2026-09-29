# SCINet

> **定位**：**下采样后再卷积**：把长序列变短以降低复杂度，递归构建 even/odd 采样树
> **任务**：`ts_forecast` | **权重**：无需预训练（PaddleTS 无官方权重，本框架用「同起点 + fp64 判定」对齐）

---

## 1. 论文与出处

| 项 | 内容 |
|---|---|
| 论文 | SCINet: Time Series Modeling and Forecasting with Sample Convolution and Interaction |
| arXiv | 2106.09305 |
| 年份 | 2021 |
| 机构 |  |
| 上游实现 | PaddleTS 1.1.0 `paddlets/models/forecasting/dl/scinet.py` |
| 本框架实现 | ``torchkiln/nn/s_scinet.py::SCINet`` |
| 对齐方式 | **同起点 dump 权重 → torch 加载 → 逐层前向 + fp64 判定** |

### 要解决的问题

长序列上直接卷积/注意力代价过高；而简单的下采样会丢信息。

### 核心创新

- **SCI 块**（Sample-Convolution-Interaction）：
  ① 拆成 **even/odd 两个子序列**（下采样 2×）
  ② 各自过卷积 → 交互（一个的输出加到另一个）
  ③ 递归多层形成**采样树**
- **复杂度从 O(L) 降到 O(L/2^k)**

---

## 2. 网络结构

### 2.1 整体框图

```
Input (B,L,D)
  └─ SCIBlock(level=1): even/odd 拆分 → 各自 Conv → 交互
       ├─ even → SCIBlock(level=2) → ...
       └─ odd  → SCIBlock(level=2) → ...
  └─ 解码器：Conv1D(in_chunk_len → out_chunk_len, k=1)
  → (B,H,D)
```

### 2.2 逐模块说明

| 模块 | 结构 | 作用 |
|---|---|---|
| `_Interactor` | Conv + 交互 | 子序列融合 |
| `_SCIBlock` | even/odd + 递归 | 采样树 |
| `_decoder1/2` | `Conv1D(in_len→out_len, k=1)` | 输出映射 |

### 2.3 与同类的差异

| 对比 | 差异 |
|---|---|
| vs TCN | TCN 膨胀扩感受野；SCINet 下采样降长度 |
| vs NHiTS | 都做多尺度，SCINet 用采样树 |

---

## 3. 输出与后处理

| 项 | 说明 |
|---|---|
| 输入 | `{"past_target": (B, L, D)}`（协变量模型另有 `known_cov_numeric` 等） |
| 输出 | `**(pred, mid_pred) 元组**（`num_stack=1` 时 mid_pred 为 None）` |
| 解码 | **直接输出，无需后处理**（时序回归不同于检测，无 NMS/anchor） |
| 指标 | MSE / MAE |

> ⚠️ 与检测任务不同：**没有 anchor、没有 NMS、没有解码** —— 网络输出即预测值。

---

## 4. 配置与用法

```yaml
Architecture:
  model_family: ts_forecast
  task: ts_forecast
  Head:
    model: scinet
    in_chunk_len: 96
    out_chunk_len: 24
    target_dim: 1
Loss:
  type: mse
Metric:
  main_indicator: MSE
  pred_mode: point
```

```bash
tkiln train  -c configs/ts/scinet_demo.yml
tkiln val    -c configs/ts/scinet_demo.yml --weights output/.../best_accuracy.pth
tkiln predict -c configs/ts/scinet_demo.yml --input data.csv
```

---

## 5. 规模与速度

| 项 | 值 |
|---|---|
| 参数 | **0.003 M** |
| 本框架对齐 | **rel **5e-6**（浮点级）** |
| 训练速度 | 合成数据 30 epoch < 1 分钟（本机 RTX 4060 Ti） |
| FLOPs | ⚠️ **本框架未集成 FLOPs 统计** |

---

## 6. 公开指标

**对齐验证**（同权重同输入，PaddleTS vs 本框架）：

| 项 | 结果 |
|---|---|
| 权重加载 | **missing=0 / unexpected=0** |
| 逐层前向 | **rel **5e-6**（浮点级）** |

> ℹ️ PaddleTS **不发布官方预训练权重**，故无"公开指标"可对比；
> 本框架的验收标准是**与 PaddleTS 参考实现数值一致**。

---

## 7. 选型建议

| 场景 | 建议 |
|---|---|
| 长序列、显存受限 | ✅ |
| 极短序列 | 下采样会过短，不适用 |

---

## 8. 参考对比

| 对比 | 说明 |
|---|---|
| 其他多尺度模型 | SCINet 的采样树是独有设计 |

---

## 9. 已知问题 / 注意事项

- ⚠️ **内层用 `paddle.nn.Pad1D`（零填充）** ⇒ torch 必须 `nn.ConstantPad1d(...,0)`，
  **不能**用 `ReplicationPad1d`
- ⚠️ 解码器在 paddle NCL 下**通道维=时间轴**，forward 直接 `decoder(x)`，勿 transpose
- ⚠️ forward 返回**元组**，框架 `tasks/ts_forecast.py::_first` 取首元素
- ⚠️ **忽略协变量**
