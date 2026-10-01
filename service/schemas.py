"""对外契约（Pydantic）。

设计要点：
  - ``JobSpec`` 是**声明式**的：平台把"跑什么"完整描述一次，服务端不做隐式补全。
    spec 原样存库（``spec_json``），因此天然可复现——TorchKiln 改了默认值也不影响
    老任务的解释，这与"从命令行反推超参"是两种东西。
  - ``params`` 用**点分键**（``{"Global.epoch_num": 100}``）而不是嵌套 dict，
    因为它就是 ``-o Key.Sub=value`` 的直接形态，转换零歧义。
  - spec 里带 ``framework_version``：服务端据此拒绝/警告版本不匹配的训练请求。
"""
from __future__ import absolute_import

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------- 作业状态
#: 终态集合（客户端可据此停止轮询）
TERMINAL = ("succeeded", "failed", "cancelled")
#: 全部状态
STATUSES = ("queued", "running") + TERMINAL

#: 作业种类。决定 :func:`service.runner.build_argv` 拼出哪条命令。
#:
#: - ``train``    ``python -m torchkiln train``（默认值，保持向后兼容）
#: - ``eval``     ``python -m torchkiln val``，需要 ``weights_path``
#:
#: 评估之所以要做成"一等作业"而不是让调用方自己拼命令：评估的**产物不是权重而是指标**，
#: 而指标只有走 ``metrics.jsonl`` 契约才能被 HTTP 消费方可靠拿到——
#: 解析控制台日志在小版本改动后会静默失效（见 :mod:`ptcore.metrics_sink` 开头的说明）。
JOB_KINDS = ("train", "eval")


class Resources(BaseModel):
    """资源诉求。单卡场景请把 ``gpu_memory_gb`` 填准，用于排队前的准入判断。"""

    gpu: int = Field(default=1, ge=0, le=16)
    gpu_memory_gb: Optional[float] = Field(default=None, gt=0)
    shm_size: Optional[str] = Field(
        default=None, description='如 "4g"；DataLoader worker 多时必须给，否则 BUS error')
    cpu: Optional[str] = Field(default=None, description='如 "4"')
    priority: int = Field(default=5, ge=0, le=9, description="数字越小越优先")


class DatasetRef(BaseModel):
    """数据集引用（容器内绝对路径）。

    由平台侧把标注导出成 TorchKiln 的布局后，把挂载点传进来即可——
    **服务不关心数据从哪来**，那是 AIStation 的职责。
    """

    data_dir: Optional[str] = None
    train_list: Optional[str] = None
    val_list: Optional[str] = None


class JobSpec(BaseModel):
    """一次作业请求的完整声明。"""

    spec_version: str = "1.0"
    framework: str = "torchkiln"
    framework_version: Optional[str] = None
    #: 作业种类。默认 ``train``，老客户端不带这个字段时行为不变。
    kind: str = "train"
    #: 二选一：模型名（走 resolve_config）或直接给配置路径
    model_name: Optional[str] = None
    config_path: Optional[str] = None
    #: 待评估权重（容器内绝对路径）。仅 ``kind="eval"`` 需要。
    weights_path: Optional[str] = None
    #: 点分键 -> 值，直接转成 ``-o k=v``
    params: Dict[str, Any] = Field(default_factory=dict)
    dataset: Optional[DatasetRef] = None
    resources: Resources = Field(default_factory=Resources)
    seed: Optional[int] = None
    #: 平台侧可塞任意透传信息（如数据集版本号、审批单号），原样保留在作业记录里
    labels: Dict[str, str] = Field(default_factory=dict)

    def validate_kind(self) -> str:
        """校验 ``kind`` 并返回归一化值。

        单独做成方法（而不是靠 pydantic 的 ``Literal``）是为了让错误信息能说清
        「有哪些可选值」，而不是只说字符串不在枚举里。
        """
        k = (self.kind or "train").strip().lower()
        if k not in JOB_KINDS:
            raise ValueError(
                "unsupported job kind: {!r}（可选：{}）".format(self.kind, ", ".join(JOB_KINDS)))
        return k


class JobCreated(BaseModel):
    job_id: str
    status: str
    #: 命中已有幂等键时为 true，此时返回的是**同一个**作业
    idempotent_hit: bool = False
    created_ts: float


class JobInfo(BaseModel):
    job_id: str
    status: str
    spec: JobSpec
    exit_reason: Optional[str] = None
    exit_code: Optional[int] = None
    created_ts: float
    started_ts: Optional[float] = None
    finished_ts: Optional[float] = None
    duration_sec: Optional[float] = None
    output_dir: Optional[str] = None
    log_path: Optional[str] = None
    metrics_path: Optional[str] = None
    #: 已落库的最后一个指标 seq，客户端可用它做断点续传
    metrics_seq: int = -1
    user_id: Optional[str] = None
    tenant: Optional[str] = None
    labels: Dict[str, str] = Field(default_factory=dict)
    #: 队列位置（仅 queued 时有意义）
    queue_position: Optional[int] = None
    error: Optional[str] = None


class JobList(BaseModel):
    total: int
    items: List[JobInfo]


class CancelResult(BaseModel):
    job_id: str
    status: str
    #: 是否由本请求触发的终止（false 表示它本来就在终态）
    terminated: bool
