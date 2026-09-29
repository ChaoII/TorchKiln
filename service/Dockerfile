# TorchKiln 训练服务镜像
#
# CUDA 版本必须与**宿主驱动**匹配。本机实测：驱动 616.56 / torch 2.14.0+cu132 /
# RTX 4060 Ti。构建时用 ARGS 覆盖：
#   docker build -f service/Dockerfile --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu132 .
#
# ⚠️ 体积提醒：CUDA runtime + torch 约 8~12GB。首次拉取慢是正常的，
#    建议 buildx 缓存或推到内网 registry，不要每次重建。
FROM nvidia/cuda:13.2.0-cudnn-runtime-ubuntu24.04

ARG TORCH_INDEX=https://download.pytorch.org/whl/cu132
ARG TORCH_VERSION=2.14.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TKILN_REPO_ROOT=/opt/torchkiln \
    TKILN_DATA_ROOT=/data/torchkiln \
    TKILN_MAX_CONCURRENT=1

# ---- 系统依赖（OCR 的 pyclipper/shapely 需要编译链；audio 的 soundfile 需要 libsndfile）
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3-pip python3.12-venv \
        git curl libsndfile1 ffmpeg \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---- torch（先装，层缓存命中率最高）
RUN python3.12 -m pip install --upgrade pip \
    && python3.12 -m pip install --index-url ${TORCH_INDEX} \
        torch==${TORCH_VERSION} torchvision

WORKDIR ${TKILN_REPO_ROOT}

# ---- 依赖层：先只拷依赖声明，改代码时这一层不失效
COPY pyproject.toml ./
COPY service/requirements.txt service/requirements.txt
RUN python3.12 -m pip install -r service/requirements.txt

# ---- 代码层
COPY ptcore/ ptcore/
COPY torchkiln/ torchkiln/
COPY configs/ configs/
COPY tools/ tools/
COPY service/ service/

# TorchKiln 本体 + 任务族可选依赖
#   [ocr]   pyclipper/shapely/albumentations/scikit-image/PyMuPDF …  (PP-OCR)
#   [audio] torchaudio/transformers/resampy/soundfile             (kokoro / panns)
#   [export] onnx/onnxruntime/onnx-slim                            (ONNX 导出)
# 不装 [audio] 也能跑（任务已改按需加载），只是用到时会明确报缺依赖。
RUN python3.12 -m pip install -e ".[ocr,export]"

# ---- 非 root 运行（多用户场景下的基本隔离）
RUN useradd -m -u 10001 trainer \
    && mkdir -p ${TKILN_DATA_ROOT} \
    && chown -R trainer:trainer ${TKILN_DATA_ROOT} ${TKILN_REPO_ROOT}
USER trainer

EXPOSE 8000

# ⚠️ 生产部署**不要**用 `--reload`，它会 fork 出子进程并自动重载，
#    与本服务的作业恢复逻辑冲突（重启会触发 recover 接管道）。
CMD ["python3.12", "-m", "uvicorn", "service.main:app", \
     "--host", "0.0.0.0", "--port", "8000"]
