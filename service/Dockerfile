# TorchKiln 训练服务镜像
#
# 基础镜像默认用官方 pytorch 运行时（已含 CUDA 13.2 runtime + cuDNN），
# 比 nvidia/cuda + 自行 pip install torch 省去数 GB 下载与编译时间。
# 切换到最小基础镜像：
#   --build-arg BASE_IMAGE=nvidia/cuda:13.2.0-cudnn-runtime-ubuntu24.04 --build-arg NEED_TORCH=1
#
# ⚠️ 体积提醒：约 10GB。首次构建慢是正常的；建议推到内网 registry 或用
#    buildx 缓存，不要每次重建。本机驱动 616.56 支持 CUDA 13.2。
ARG BASE_IMAGE=pytorch/pytorch:2.13.0-cuda13.2-cudnn9-runtime
FROM ${BASE_IMAGE}

# 是否自行安装 torch（用 nvidia/cuda 基础镜像时置 1）
ARG NEED_TORCH=0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu132
ARG TORCH_VERSION=2.14.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TKILN_REPO_ROOT=/opt/torchkiln \
    TKILN_DATA_ROOT=/data/torchkiln \
    TKILN_MAX_CONCURRENT=1

# ---- 系统依赖 ----
# libsndfile1: audio 族的 soundfile；ffmpeg: 音视频解码
# build-essential: pyclipper/shapely 需要编译链
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates \
        libsndfile1 ffmpeg \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---- torch（仅当用 nvidia/cuda 基础镜像时）----
RUN if [ "${NEED_TORCH}" = "1" ]; then \
        python -m pip install --upgrade pip && \
        python -m pip install --index-url ${TORCH_INDEX} torch==${TORCH_VERSION} torchvision ; \
    fi

WORKDIR ${TKILN_REPO_ROOT}

# ---- 依赖层：先只拷声明文件，改代码时这一层不失效 ----
COPY pyproject.toml ./
COPY service/requirements.txt service/requirements.txt
RUN python -m pip install -r service/requirements.txt

# ---- 代码层 ----
COPY ptcore/ ptcore/
COPY torchkiln/ torchkiln/
COPY configs/ configs/
COPY tools/ tools/
COPY service/ service/

# TorchKiln 本体 + 任务族可选依赖
#   [ocr]   pyclipper/shapely/albumentations/scikit-image/PyMuPDF …  (PP-OCR)
#   [audio] torchaudio/transformers/resampy/soundfile             (kokoro/panns)
#   [export] onnx/onnxruntime/onnx-slim                            (ONNX 导出)
#
# ⚠️ albumentations 钉 1.4.24：2.x 依赖 albucore→simsimd→numkong，
#    0.4.x 又依赖 imgaug（会拖 OpenCV 版本），1.4.24 是依赖链最浅且
#    满足 torchkiln/ocr/data/imaug/iaa_augment.py 所需 API 的版本。
RUN python -m pip install -e ".[ocr,export]" \
    && python -m pip install "albumentations==1.4.24" "albucore==0.0.24" simsimd stringzilla \
    && python -c "import torch, torchkiln.cli; print('torch', torch.__version__)"

# ---- 自检：OCR / YOLO 两条通路都能构建（缺依赖会在镜像里才暴露，代价很高）----
RUN python -m torchkiln list > /dev/null \
    && python -c "import torchkiln.tasks as T; print('tasks:', len(T.AVAILABLE))"

# ---- 非 root 运行（多用户场景的基本隔离）----
RUN useradd -m -u 10001 trainer \
    && mkdir -p ${TKILN_DATA_ROOT} \
    && chown -R trainer:trainer ${TKILN_DATA_ROOT} ${TKILN_REPO_ROOT}
USER trainer

EXPOSE 8000

# ⚠️ 生产**不要**加 --reload：它会 fork 子进程并自动重载，与本服务的
#    作业恢复逻辑冲突（重启会触发 recover 接管道）。
CMD ["python", "-m", "uvicorn", "service.main:app", \
     "--host", "0.0.0.0", "--port", "8000"]
