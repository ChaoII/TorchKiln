# TorchKiln 训练服务镜像
#
# ⚠️ 不要加 `# syntax=docker/dockerfile:1`：它会让 Docker 去 registry-1.docker.io
#    拉 frontend 镜像，国内网络基本拉不通，构建直接失败（实测报
#    "failed to resolve source metadata for docker.io/docker/dockerfile:1"）。
#    本文件只用内置指令，不需要额外 frontend。
#
# 基础镜像默认用官方 pytorch 运行时（已含 CUDA 13.2 runtime + cuDNN），
# 比 nvidia/cuda + 自行 pip install torch 省去数 GB 下载与编译时间。
# 切换到最小基础镜像：
#   --build-arg BASE_IMAGE=nvidia/cuda:13.2.0-cudnn-runtime-ubuntu24.04 --build-arg NEED_TORCH=1
#
# ⚠️ 体积提醒：约 10GB。本机驱动 616.56 支持 CUDA 13.2。
#
# ==============================================================
# ⚠️ 源一律走国内镜像：apt 用**阿里云**，pip 用**清华**。
#    走官方源会慢到无法忍受（实测从 archive.ubuntu.com 拉 apt 依赖，
#    5 分钟只走了 85 个包还没进下一步）。
#    换源用 --build-arg 覆盖，**不要**改回官方源。详见 AGENTS.md。
# ==============================================================
ARG BASE_IMAGE=pytorch/pytorch:2.13.0-cuda13.2-cudnn9-runtime
FROM ${BASE_IMAGE}

# ⚠️ 必须在 FROM **之后**重新声明：FROM 之前声明的 ARG 只在 FROM 指令里可见，
#    构建阶段拿不到（会导致 ${APT_MIRROR} 展开成空，apt 源写坏）。
ARG APT_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/ubuntu
ARG PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_HOST=mirrors.tuna.tsinghua.edu.cn
# PyTorch CUDA 轮子的国内镜像。**实测测速结论（2026-09-29，勿凭印象换源）**：
#   清华   https://mirrors.tuna.tsinghua.edu.cn/pytorch-wheels/cu132/   ✅ 6.2 MB/s
#   中科大 https://mirrors.ustc.edu.cn/pytorch-wheels/cu132/           ✅ 5.2 MB/s
#   阿里云 https://mirrors.aliyun.com/pytorch-wheels/cu132/             ⚠️ 0.02 MB/s（82s 拉 1.7MB），别用
#   南科大 https://mirrors.sustech.edu.cn/pytorch-wheels/cu132/         ❌ 反爬挑战，pip 拉不到
#   南京大 https://mirror.nju.edu.cn/pytorch/whl/cu132/                  ⚠️ 无 torch 2.1x 条目
# ⚠️ 测速必须 GET 拉真实字节；`Invoke-WebRequest -Method Head` 返回 200 **不可信**
#    （南科大就是反爬典型：HEAD 200、GET 返回验证页）。
ARG TORCH_WHEEL_INDEX=https://mirrors.tuna.tsinghua.edu.cn/pytorch-wheels/cu132/
ARG NEED_TORCH=0
ARG TORCH_VERSION=2.14.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=${PIP_INDEX} \
    PIP_TRUSTED_HOST=${PIP_HOST} \
    PIP_DEFAULT_TIMEOUT=60 \
    PIP_RETRIES=5 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    TKILN_REPO_ROOT=/opt/torchkiln \
    TKILN_DATA_ROOT=/data/torchkiln \
    TKILN_MAX_CONCURRENT=1

# ⚠️ `PIP_BREAK_SYSTEM_PACKAGES=1` 不是可选的：Ubuntu 24.04 与
#    pytorch/pytorch 镜像的系统 Python **都是 PEP 668 externally-managed**
#    （实测两者都存在 EXTERNALLY-MANAGED 标记），直接 `pip install` 会报
#    "This environment is externally managed" 并退出码 1。
#    容器里是一次性环境，允许装进系统 Python；该变量对未受管的 Python 无副作用。

# ---- 国内 apt 源（清华，实测 6.2 MB/s）----
# ⚠️ Ubuntu 24.04(noble) 用 deb822 格式的 /etc/apt/sources.list.d/ubuntu.sources，
#    只改 sources.list 不够——必须一并删掉，否则 apt 同时读两个源、照样访问官方站。
# ⚠️ 用 echo 而不是 printf '\n'：反斜杠 n 会被 Dockerfile 当成真换行，
#    把单引号字符串拆到两行、shell 语法直接崩（exit code 2）。
# ⚠️ 必须 `set -eux`：没有 -e 时中间步骤失败会被后续步骤掩盖，
#    报错指向无关的地方，排查成本极高。
RUN set -eux; \
    rm -f /etc/apt/sources.list \
          /etc/apt/sources.list.d/ubuntu.sources \
          /etc/apt/sources.list.d/*.sources \
          /etc/apt/sources.list.d/*.list; \
    echo "deb ${APT_MIRROR} noble main restricted universe multiverse" >  /etc/apt/sources.list; \
    echo "deb ${APT_MIRROR} noble-updates main restricted universe multiverse" >> /etc/apt/sources.list; \
    echo "deb ${APT_MIRROR} noble-backports main restricted universe multiverse" >> /etc/apt/sources.list; \
    echo "deb ${APT_MIRROR} noble-security main restricted universe multiverse" >> /etc/apt/sources.list; \
    cat /etc/apt/sources.list

# ---- 系统依赖 ----
# libsndfile1: audio 族的 soundfile；ffmpeg: 音视频解码
# build-essential: pyclipper/shapely 需要编译链
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        git curl ca-certificates \
        libsndfile1 ffmpeg \
        build-essential; \
    rm -rf /var/lib/apt/lists/*

# ---- torch（仅当用 nvidia/cuda 基础镜像时）----
# ⚠️ 走国内 PyTorch 轮子镜像，不要用 download.pytorch.org（国内极慢）。
#    实测结论见上面 ARG 处的注释：阿里云可用，南科大被反爬挡着。
RUN set -eux; \
    if [ "${NEED_TORCH}" = "1" ]; then \
        python -m pip install --upgrade pip; \
        python -m pip install --index-url "${TORCH_WHEEL_INDEX}" \
            "torch==${TORCH_VERSION}" torchvision; \
    fi

# ---- 构建身份：让服务能自报「镜像是从哪份代码构建的」------------------
# 这样平台侧能在提交作业**之前**判断镜像是否过期，而不是等提交后收到一个
# 「404 Not Found」或「AttributeError: MetricSink 没有 predict」这类指向不明
# 的报错。用 service/build-image.ps1 构建会自动填上当前 git 修订号。
ARG TKILN_REVISION=unknown
ARG TKILN_DIRTY=false
ENV TKILN_REVISION=${TKILN_REVISION}\
    TKILN_DIRTY=${TKILN_DIRTY}\
    TKILN_REPO_ROOT=${TKILN_REPO_ROOT}

WORKDIR ${TKILN_REPO_ROOT}

# ---- 依赖层：先只拷声明文件，改代码时这一层不失效 ----
COPY pyproject.toml ./
COPY service/requirements.txt service/requirements.txt
RUN set -eux; python -m pip install -r service/requirements.txt

# ---- 代码层 ----
COPY ptcore/ ptcore/
COPY torchkiln/ torchkiln/
COPY configs/ configs/
COPY tools/ tools/
COPY service/ service/

# TorchKiln 本体 + 任务族可选依赖
#   [ocr]   pyclipper/shapely/albumentations/scikit-image/PyMuPDF …  (PP-OCR)
#   [audio] torchaudio/transformers/resampy/soundfile             (kokoro/panns)
#   [export] onnx/onnxruntime/onnxslim                           (ONNX 导出)
#
# ⚠️ albumentations 钉 1.4.24：2.x 依赖 albucore→simsimd→numkong（越装越深），
#    0.4.x 又依赖 imgaug（会拖 OpenCV 版本），1.4.24 是依赖链最浅且满足
#    torchkiln/ocr/data/imaug/iaa_augment.py 所需 API 的版本。
# ⚠️ **只钉 albumentations 本身，不要顺手钉 albucore/simsimd/stringzilla**：
#    它们是 albumentations 的依赖，让 pip 自己解析即可。手动钉版本会和
#    resolver 选出的版本冲突，报
#    "Cannot install albucore==0.0.24 and albumentations==1.4.24 because
#     these package versions have conflicting dependencies"。
RUN set -eux; \
    python -m pip install -e ".[ocr,export]"; \
    python -m pip install "albumentations==1.4.24"; \
    python -c "import torch, torchkiln.cli; print('torch', torch.__version__)"

# ---- 自检：缺依赖会在镜像里才暴露，代价很高，这里提前拦 ----
RUN set -eux; \
    python -m torchkiln list > /dev/null; \
    python -c "import torchkiln.tasks as T; print('tasks:', len(T.AVAILABLE))"

# ---- 非 root 运行（多用户场景的基本隔离）----
RUN set -eux; \
    useradd -m -u 10001 trainer; \
    mkdir -p "${TKILN_DATA_ROOT}"; \
    chown -R trainer:trainer "${TKILN_DATA_ROOT}" "${TKILN_REPO_ROOT}"
USER trainer

EXPOSE 8000

# ⚠️ 生产**不要**加 --reload：它会 fork 子进程并自动重载，与本服务的
#    作业恢复逻辑冲突（重启会触发 recover 接管道）。
CMD ["python", "-m", "uvicorn", "service.main:app", \
     "--host", "0.0.0.0", "--port", "8000"]
