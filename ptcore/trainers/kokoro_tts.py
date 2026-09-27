"""kokoro TTS trainer (``Architecture.task: kokoro_tts``)。

共享训练循环在 :class:`ptcore.trainers.base.BaseTrainer`；
任务相关的模型/损失/指标/数据集构造在 ``torchkiln/tasks/kokoro_tts.py``。

⚠️ 已知限制（见 AGENTS）：kokoro 的 ``Generator`` 含随机噪声；
**整链一次反传在部分输入下会触发原生段错误（0xC0000005）** ——
smoke 与 CLI 冒烟均已验证常规 batch 可正常反传，若遇崩溃需**按模块分段反传**。
"""
from __future__ import absolute_import

from ptcore.trainers.base import BaseTrainer

__all__ = ["KokoroTtsTrainer"]


class KokoroTtsTrainer(BaseTrainer):
    """kokoro-82M 文本转语音（TTS）。"""

    def _default_task(self, config):
        from torchkiln.tasks import get_task

        return get_task("kokoro_tts")
