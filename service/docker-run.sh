# TorchKiln 训练服务 —— 容器运行
#
# 前提：宿主机已装 NVIDIA 驱动 + nvidia-container-toolkit
#   docker info | grep -i nvidia     # 需能看到 nvidia runtime
#
# 单卡训练：把并发设为 1（显存不够会 OOM）
# 多卡：见下方 --gpus 与 TKILN_MAX_CONCURRENT 说明

set -euo pipefail

IMAGE="${IMAGE:-torchkiln:0.1.0}"
PORT="${PORT:-8000}"
# ⚠️ 单卡务必 1：RTX 4060 Ti 只有 16GB，并发 2 会 OOM
CONCURRENCY="${CONCURRENCY:-1}"
TOKEN="${TOKEN:?必须设置 TKILN_SERVICE_TOKEN，勿用公开默认值}"
DATA_DIR="${DATA_DIR:-$(pwd)/tk-data}"

mkdir -p "${DATA_DIR}"

# 产物/指标/权重都落这里（宿主可见）——指标真值 metrics.jsonl 也在其中
docker run -d --name torchkiln \
  --gpus all \
  --shm-size=8g \
  --restart unless-stopped \
  -p "${PORT}:8000" \
  -e TKILN_SERVICE_TOKEN="${TOKEN}" \
  -e TKILN_MAX_CONCURRENT="${CONCURRENCY}" \
  -e TKILN_POLL_INTERVAL=1.0 \
  -e TKILN_REPO_ROOT=/opt/torchkiln \
  -e TKILN_DATA_ROOT=/data/torchkiln \
  -v "${DATA_DIR}:/data/torchkiln" \
  "${IMAGE}" \
  python -m uvicorn service.main:app --host 0.0.0.0 --port 8000

echo "started; health check:"
for i in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:${PORT}/healthz" >/dev/null 2>&1; then
    curl -fsS "http://127.0.0.1:${PORT}/healthz"; echo
    exit 0
  fi
  sleep 2
done

echo "health check failed; last 50 log lines:" >&2
docker logs --tail 50 torchkiln >&2
exit 1
