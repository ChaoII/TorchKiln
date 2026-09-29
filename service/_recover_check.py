"""重启恢复验证：服务重启时进行中的训练必须被接管，不能变僵尸。

流程：
  1. 起服务，提交一个长作业，等它 running
  2. **杀掉服务**（模拟崩溃/升级）
  3. 重起服务，检查该作业：应仍为 running，且能继续读到指标
  4. 训练自然结束后，作业应自动收敛为 succeeded

用法: python service/_recover_check.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request

PY = r"C:\Users\aichao\.conda\envs\ultraly\python.exe"
REPO = r"D:\TorchKiln"
PORT = 8778
BASE = "http://127.0.0.1:{}".format(PORT)
DATA = os.path.join(REPO, "output", "_svc_recover")
HDR = {"Authorization": "Bearer rectest", "Content-Type": "application/json"}
ENV = dict(os.environ, TKILN_PYTHON=PY, TKILN_REPO_ROOT=REPO, TKILN_DATA_ROOT=DATA,
           TKILN_SERVICE_TOKEN="rectest", TKILN_POLL_INTERVAL="0.5", PYTHONPATH=REPO)


def kill_stale():
    """清掉上次异常退出残留的 8778 服务——否则 start() 会连上**别人**的旧进程，
    测出来的"恢复"全是假的。"""
    killed = 0
    ps = (
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        "Where-Object { $_.CommandLine -like '*" + str(PORT) + "*' } | "
        "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True)
    time.sleep(2)
    try:
        call("/healthz")
        print("  !! 端口 {} 仍被占用，测试无意义".format(PORT))
        sys.exit(2)
    except Exception:
        pass
    return killed


def call(path, body=None, method=None, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data,
                                 headers=headers or HDR, method=method or ("POST" if data else "GET"))
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def start():
    """起服务。

    ⚠️ stderr **必须**重定向到文件，不能用 PIPE：没人读 PIPE 时，管道缓冲区一满，
    uvicorn 会在事件循环里同步写 stderr 而把整个 loop 堵死——表现为
    「训练进程在跑、日志/metrics 一点不动」，极难排查。
    """
    errlog = open(os.path.join(DATA + ".err.log"), "ab")
    p = subprocess.Popen(
        [PY, "-m", "uvicorn", "service.main:app", "--host", "127.0.0.1",
         "--port", "8778", "--log-level", "warning"],
        cwd=REPO, env=ENV, stdout=errlog, stderr=errlog)
    for _ in range(60):
        time.sleep(1)
        try:
            call("/healthz")
            return p
        except Exception:
            if p.poll() is not None:
                with open(DATA + ".err.log", "rb") as f:
                    print(f.read().decode(errors="replace")[-1500:])
                raise
    raise RuntimeError("service did not start")


def main():
    print("== 步骤0：清理陈旧进程与数据 ==")
    kill_stale()
    subprocess.run(["cmd", "/c", "rmdir", "/s", "/q", DATA], capture_output=True)
    time.sleep(1)
    print("  已清理")

    print("\n== 步骤1：起服务 + 提交长作业 ==")
    proc = start()
    jid = call("/api/v1/train/jobs",
               {"model_name": "yolov8n-det",
                "params": {"Global.epoch_num": 8, "Global.print_batch_step": 1}},
               headers=dict(HDR, **{"Idempotency-Key": "recover-{}".format(time.time())}))["job_id"]
    print("  job_id =", jid)
    # 输出目录必须存在，metrics.jsonl 才会被 tail
    jdir = os.path.join(DATA, "jobs", jid)
    for _ in range(90):
        time.sleep(1)
        info = call("/api/v1/train/jobs/" + jid)
        if info["status"] == "running" and os.path.isfile(
                os.path.join(jdir, "metrics.jsonl")):
            break
    print("  status =", info["status"], "| metrics.jsonl 已生成:",
          os.path.isfile(os.path.join(jdir, "metrics.jsonl")))
    # 等到真的产出指标事件再杀，否则测不到"游标连续性"这个关键点
    for _ in range(60):
        time.sleep(1)
        t0 = time.time()
        call("/healthz")                     # 探活：确认事件循环没被堵
        rt = time.time() - t0
        before = call("/api/v1/train/jobs/" + jid)
        if before["metrics_seq"] >= 0:
            print("  /healthz 往返 {:.0f} ms（事件循环通畅）".format(rt * 1000))
            break
    seq_before = before["metrics_seq"]
    mpre = call("/api/v1/train/jobs/{}/metrics?offset=-1".format(jid))["items"]
    print("  重启前 metrics_seq = {}，已产出 {} 条事件".format(seq_before, len(mpre)))
    assert seq_before >= 0, "杀进程前应已产出指标"

    print("\n== 步骤2：强杀服务（模拟崩溃）==")
    proc.kill()
    proc.wait(timeout=30)
    time.sleep(2)
    try:
        call("/healthz")
        print("  !! 服务还活着，异常")
        return False
    except Exception:
        print("  服务已停止")

    print("\n== 步骤3：重启服务（模拟升级后恢复）==")
    proc = start()
    time.sleep(3)
    after = call("/api/v1/train/jobs/" + jid)
    print("  重启后 status =", after["status"], "（期望 running，不能是 failed/queued）")
    recovered = after["status"] == "running"
    m = call("/api/v1/train/jobs/{}/metrics?offset=-1".format(jid))
    print("  历史指标可读 =", len(m["items"]), "条（>= 重启前的 {} 条）".format(len(mpre)))
    print("  历史指标未丢失 =", len(m["items"]) >= len(mpre))
    lost = after["metrics_seq"] < seq_before
    print("  指标游标未回退 =", not lost, "({} -> {})".format(seq_before, after["metrics_seq"]))
    grew = after["metrics_seq"] > seq_before
    print("  接管后游标继续前进 =", grew)

    print("\n== 步骤4：等训练自然结束，应自动收敛 ==")
    final = None
    for _ in range(600):
        time.sleep(3)
        final = call("/api/v1/train/jobs/" + jid)
        if final["status"] in ("succeeded", "failed", "cancelled"):
            break
    print("  终态: status={} exit_reason={} 耗时={}s".format(
        final["status"], final["exit_reason"], final["duration_sec"]))
    converged = final["status"] == "succeeded"

    proc.kill()
    ok = recovered and not lost and grew and len(m["items"]) >= len(mpre) and converged
    print("\n===== 重启恢复: {} =====".format("PASS" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
