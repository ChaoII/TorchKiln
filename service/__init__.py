"""TorchKiln 训练服务（对外 HTTP 契约层）。

分层：
    settings   环境变量配置
    schemas    对外契约（Pydantic）
    store      作业状态持久化（SQLite，重启可恢复）
    streams    SSE 广播 + 断点续传
    runner     JobSpec -> 训练进程 + 指标消费 + 终态判定
    jobs       队列/幂等/取消/重启恢复
    registry   模型自描述（包 ptcore.config_schema）
    main       FastAPI 装配

⚠️ 分工原则：**指标真值永远是 ``<output_dir>/metrics.jsonl``**，本服务只做转发，
不做二次存储；GPU 排队只在服务端发生（AIStation 侧只做提交节流）。
"""
from __future__ import annotations
