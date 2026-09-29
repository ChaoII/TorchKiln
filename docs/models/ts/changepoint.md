# 变点 / 概念漂移检测

> **定位**：**纯统计工具**（无模型、无训练）：流式监控数据是否发生结构性变化
> **任务**：`工具（`tkiln changepoint`）` | **权重**：无需权重（纯算法）

---

## 1. 论文与出处

| 项 | 内容 |
|---|---|
| 论文 | CUSUM (Page 1954) / Page-Hinkley (1971) / ADWIN (Bifet & Gavaldà 2007) |
| arXiv | — |
| 年份 | 1954~2007 |
| 上游实现 | 统计过程控制（SPC）/ 在线学习文献 |
| 本框架实现 | `torchkiln/ts_changepoint.py + tools/changepoint.py + cli.py::TOOL_SCRIPTS` |
| 对齐方式 | **无需对齐**（纯 numpy 实现） |

### 要解决的问题

持续数据流中，**分布何时发生变化**（设备老化、工况切换、数据源漂移）？需要**在线、O(1)、无需训练**的检测。

### 核心创新

4 个业界常用算法：
| 算法 | 原理 | 擅长 |
|---|---|---|
| **CUSUM** | `S=max(0, S+z-δ)` 累积偏离 | **均值微小持续偏移** |
| **Page-Hinkley** | `PH = m_t - min(m)` | 在线漂移，O(1) |
| **ADWIN** | 窗口二分 + Hoeffding 界 | **概念漂移** |
| **z-score** | 滚动均值/标准差 | **突变/离群** |

---

## 2. 网络结构

```
Stream: x_1, x_2, x_3, ...
  └─ Detector.update(x) -> bool（True = 触发变点）
       · det.last_change_index  变点位置
       · det.scores             逐点分数

离线: detect_changepoints(arr, method='cusum', **kw)
      -> {'indices': [...], 'scores': ndarray, 'method': str}
```

| 类 | 关键参数 | 说明 |
|---|---|---|
| `CUSUMDetector` | `delta`(松弛) `threshold` | 双侧累积和 |
| `PageHinkleyDetector` | `delta` `threshold` | 在线漂移 |
| `ADWINDetector` | `delta` `max_window` | 自适应窗口 |
| `ZScoreDetector` | `window` `threshold` | 突变 |
| `detect_changepoints` | `method` `ref_frac` | 离线批量接口 |

⚠️ **CLI 注册方式特殊**：`tkiln changepoint` 走 `cli.py::TOOL_SCRIPTS` + `runpy`（**非 task**）。
⚠️ **`argv` 必须在分支之前归一化**（否则判断失效，曾导致命令不生效）。

---

## 3. 输出与后处理

| 项 | 说明 |
|---|---|
| 流式 | `det.update(x) -> bool`（O(1)） |
| 离线 | `detect_changepoints(arr) -> {indices, scores, method}` |
| CLI 输出 | JSON：`n_changepoints` / `changepoints` / `score_max` |
| 无 anchor/NMS | 纯数值算法 |

> ⚠️ 时序任务**没有 anchor / NMS / 解码**（那是检测任务的概念）；
> 但**异常检测有「阈值判定」**、**RUL 有「分段标签」** —— 见上表。

---

## 4. 配置与用法

```yaml
# 无需 task 配置；CLI 直接用：
# Detect:
#   csv_path: datasets/xxx/data.csv
#   data_dir: datasets/xxx
#   method: cusum
#   threshold: 8
#   ref_frac: 0.2
```

```bash
tkiln changepoint datasets/ts_anomaly_demo/data.csv --method cusum --threshold 8 --ref-frac 0.2
tkiln changepoint -c configs/ts/changepoint_demo.yml
```

---

## 5. 规模与速度

| 项 | 值 |
|---|---|
| 参数 | ****0（纯算法，无参数）**** |
| 对齐 | **无需对齐（纯 numpy）** |
| 本机速度 | **O(1) 在线**；离线 1200 点 < 1 秒 |
| FLOPs | ⚠️ 本框架未集成 FLOPs 统计 |

---

## 6. 公开指标

| 算法 | 检出数 | 命中 | 平均延迟 |
|---|---|---|---|
| **cusum** | 176 | **3/3** | **1 点** ⭐ |
| page_hinkley | 57 | **3/3** | 7 点 |
| adwin | 6 | **3/3** | 8 点 |
| zscore | 4 | 2/3（**漏缓漂移**） | 0 点 |

> 合成 1200 点，真实变点 400/700/900

---

## 7. 选型建议

| 场景 | 建议 |
|---|---|
| **均值缓慢漂移**（设备老化） | ✅ **CUSUM**（延迟 1 点） |
| **突变/离群** | ✅ z-score |
| **概念漂移**（分布变） | ✅ ADWIN |

---

## 8. 参考对比

| 对比 | 说明 |
|---|---|
| vs `ts_anomaly` | 异常检测找**单点异常**；变点检测找**分布变化点** |
| vs 模型类方法 | 纯统计，**无需训练/GPU** |

---

## 9. 已知问题 / 注意事项

- ⚠️ **`threshold` 需先标定**：用「正常」数据跑 `init_baseline()` 或调 `ref_frac`
- ⚠️ `cusum` 的 `delta` 太小会频繁误报
- ⚠️ **CLI 的 `argv` 必须在 `TOOL_SCRIPTS` 分支之前归一化**（已修）
- ⚠️ 工业告警建议**组合使用**
