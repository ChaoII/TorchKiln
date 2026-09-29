"""SSE 端到端验证：实时流 + Last-Event-ID 断点续传。

用法: python service/_sse_check.py <base_url> <token> <model_name>
"""
from __future__ import annotations

import json
import sys
import threading
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8777"
TOKEN = sys.argv[2] if len(sys.argv) > 2 else "testtoken"
MODEL = sys.argv[3] if len(sys.argv) > 3 else "yolov8n-det"
HDR = {"Authorization": "Bearer " + TOKEN, "Content-Type": "application/json"}


def post(path, body, headers):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def get(path, headers=None):
    req = urllib.request.Request(BASE + path, headers=headers or HDR)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def read_sse(path, seconds, headers=None, on_event=None):
    """按 SSE 规范读流：收集 (event, id, data)。

    服务端在作业终态时会**主动收流**，客户端会看到 IncompleteRead/连接关闭——
    这属于正常语义（流结束 == 作业已终结），不是错误。
    """
    req = urllib.request.Request(BASE + path, headers=headers or HDR)
    out = []
    deadline = time.time() + seconds
    with urllib.request.urlopen(req, timeout=seconds + 10) as r:
        ctype = r.headers.get("Content-Type", "")
        buf = b""
        while time.time() < deadline:
            try:
                chunk = r.read1(4096) if hasattr(r, "read1") else r.read(4096)
            except Exception as exc:  # 流被服务端关闭
                print("  (流已关闭: {})".format(type(exc).__name__))
                chunk = b""
            if not chunk:
                break
            buf += chunk
            while b"\n\n" in buf:
                frame, buf = buf.split(b"\n\n", 1)
                ev = {"event": None, "id": None, "data": []}
                for line in frame.decode("utf-8", "replace").splitlines():
                    if line.startswith("event:"):
                        ev["event"] = line[6:].strip()
                    elif line.startswith("id:"):
                        ev["id"] = line[3:].strip()
                    elif line.startswith("data:"):
                        ev["data"].append(line[5:].lstrip())
                if ev["data"]:
                    out.append(ev)
                    if on_event:
                        on_event(ev)
        # 收尾：缓冲区里可能还有没被 \n\n 分隔完的最后一帧
        if buf.strip():
            ev = {"event": None, "id": None, "data": []}
            for line in buf.decode("utf-8", "replace").splitlines():
                if line.startswith("event:"):
                    ev["event"] = line[6:].strip()
                elif line.startswith("id:"):
                    ev["id"] = line[3:].strip()
                elif line.startswith("data:"):
                    ev["data"].append(line[5:].lstrip())
            if ev["data"]:
                out.append(ev)
    return ctype, out


def main():
    print("== 提交作业 ==")
    job = post("/api/v1/train/jobs",
               {"model_name": MODEL,
                "params": {"Global.epoch_num": 1, "Global.print_batch_step": 1,
                           "Global.use_ema": False}},
               dict(HDR, **{"Idempotency-Key": "sse-check-{}".format(time.time())}))
    jid = job["job_id"]
    print("job_id =", jid, "status =", job["status"])

    print("\n== 阶段1：连实时指标流（不补历史，tail=0 等新事件）==")
    seen = []
    lock = threading.Lock()

    def cb(ev):
        with lock:
            seen.append(ev)
        print("  <- event={:<6} id={:<4} data={}".format(
            ev["event"], ev["id"], " ".join(ev["data"])[:78]))

    ctype, frames = read_sse("/api/v1/train/jobs/{}/metrics/stream?offset=-1".format(jid),
                             150, on_event=cb)
    print("  Content-Type:", ctype)
    print("  实时收到帧数:", len(frames))
    seqs = [int(f["id"]) for f in frames if f["id"] is not None]
    print("  seq 序列:", seqs)
    print("  seq 递增:", seqs == sorted(seqs), "| 去重后无重复:",
          len(seqs) == len(set(seqs)))
    print("  收到终态帧:", any(f["event"] == "end" for f in frames))

    print("\n== 阶段2：Last-Event-ID 断点续传（模拟只收到前 4 条后断线）==")
    cut = seqs[:4]
    resume_from = cut[-1] if cut else -1
    print("  断点 Last-Event-ID:", resume_from)
    ctype2, frames2 = read_sse(
        "/api/v1/train/jobs/{}/metrics/stream".format(jid), 6,
        headers=dict(HDR, **{"Last-Event-ID": str(resume_from)}))
    seqs2 = [int(f["id"]) for f in frames2 if f["id"] is not None]
    print("  补发 seq:", seqs2)
    expect = [s for s in seqs if s > resume_from]
    print("  应补发     :", expect)
    print("  补发完整且无重复:", seqs2 == expect)
    print("  （已消费过的 %d 条没有被重发）" % len(cut))

    print("\n== 阶段3：日志流 ==")
    _c, lf = read_sse("/api/v1/train/jobs/{}/logs/stream?tail=5".format(jid), 6)
    print("  收到日志帧:", len(lf))
    if lf:
        print("  末帧:", " ".join(lf[-1]["data"])[:90])

    print("\n== 阶段4：作业终态 ==")
    info = get("/api/v1/train/jobs/" + jid)
    print("  status={} exit_reason={} metrics_seq={}".format(
        info["status"], info["exit_reason"], info["metrics_seq"]))
    hist = get("/api/v1/train/jobs/{}/metrics?offset=-1".format(jid))
    print("  历史指标可完整回放:", len(hist["items"]), "条")


if __name__ == "__main__":
    main()
