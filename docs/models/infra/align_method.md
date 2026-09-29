# 对齐方法论（复现上游的验收标准）

> **定位**：与上游（ultralytics / PaddleTS / Paddle3D / PaddleSpeech / PaddleOCR）对齐的**统一方法**
> **分类**：infra

---

## 五条验收标准

| # | 项 | 判据 |
|---|---|---|
| ① | **权重加载** | `missing=0 / unexpected=0`（或仅差函数式 DFL） |
| ② | **同权重逐层前向** | maxdiff ~1e-5 |
| ③ | **单步 loss / 梯度** | loss 相对误差 <1e-3；梯度仅算子级微差 |
| ④ | **同权重推理指标** | 差 ≤0.02 |
| ⑤ | **端到端训练** | 终值差 ≤0.02~0.03（含随机性） |

> **1-4 项对齐即复现成功**；第 5 项天生有噪声。
---

## ⭐ fp64 判定法（最有用）

fp32 对拍时若 rel 只有 1e-4，**无法判断**是「fp32 舍入」还是「真 bug」。

**做法**：两侧都切 fp64 再对拍
- fp64 rel → **1e-11 级** ⇒ 证明是舍入，**移植正确**
- fp64 仍 1e-4 ⇒ 才是真 bug

**实例**：
- PANNs fp32 rel 6.8e-4 → **fp64 3.55e-11** ⇒ 正确
- Paddle GPU fp32 matmul 走 **TF32**：MLP 路径 fp32 差 4e-4 → fp64 塌到 **1e-13** ⇒ 正确
- MDTC 特征端 1.09%：**fp64 逐位不变** ⇒ 确证**不是舍入**，是两版 kaldi 移植差异
---

## 跨框架对拍的 4 个易错点

1. **`Tensor.max(axis)`**：Paddle 只返回值，torch 返回 `(values, idx)` ⇒ 用 `torch.amax`
2. **Linear 权重布局**：Paddle `[in,out]` vs torch `[out,in]` ⇒ **对比前转置**
   （⚠️ 方形 Linear 形状反转检测会失效 ⇒ **识别出 nn.Linear 后无条件转置**）
3. **Paddle BN**：`_mean/_variance` → `running_mean/var`，momentum 0.9 等价 torch 0.1
4. **Paddle RNN 默认 `time_major=False`**（batch-first）⇒ torch 必须 `batch_first=True`
---

## 零梯度参数必须用绝对判据

ECAPA 的 `asp.conv.conv.bias` 其 fp64 梯度 **8.88e-16**
（**数学上恒 0** —— ASP.conv 的输出进 softmax，bias 的加性常数被 softmax 平移不变性抵消）。

用 `rel = diff / max|grad|` 会因分母过小**虚高到 8.5** ⇒ 误判 FAIL。
**正确做法**：分母 < 1e-10 时改用**绝对误差判据**。
---

## 含随机性模块的处理

| 情形 | 做法 |
|---|---|
| 模型含随机噪声（kokoro Generator） | **两侧同步 `torch.manual_seed`** |
| 含随机采样（Informer ProbSparse） | 承认**不可逐位**，只验证前向形状 / 结构 |
| 数据增强含随机 | 对拍时**关闭增广**或注入固定 mask |
| dropout | 对拍时 `eval()` 或 `dropout=0` |
---

## 对比时的口径纪律

- ⚠️ **必须用同一评估器评双方权重**（ultra 自报 mAP 默认 `rect=True` 会抬高）
- ⚠️ **必须用同一 val 清单**（框架 val50 vs ultra val402 不可比）
- ⚠️ **单步对比必须加载完全相同的权重**（曾因 miss=102 vs miss=594 误判）
- ⚠️ **读 ultra `results.csv` 注意列序**：`mAP50-95(B)` 是 index 10、`(M)` 是 index 14
