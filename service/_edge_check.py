"""边界路径验证：取消 + setup 失败（无 end 事件）+ 队列串行。

用法: python service/_edge_check.py <base_url> <token>
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8777"
TOKEN = sys.argv[2] if len(sys.argv) > 2 else "testtoken"
HDR = {"Authorization": "Bearer " + TOKEN, "Content-Type": "application/json"}


def call(path, body=None, method=None, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data,
                                 headers=headers or HDR, method=method or ("POST" if data else "GET"))
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def wait_terminal(jid, timeout=240):
    end = time.time() + timeout
    while time.time() < end:
        info = call("/api/v1/train/jobs/" + jid)
        if info["status"] in ("succeeded", "failed", "cancelled"):
            return info
        time.sleep(2)
    return call("/api/v1/train/jobs/" + jid)


def submit(spec, key):
    return call("/api/v1/train/jobs", spec,
                headers=dict(HDR, **{"Idempotency-Key": key}))["job_id"]


def case_cancel():
    print("== 用例1：运行中取消 ==")
    jid = submit({"model_name": "yolov8n-det",
                  "params": {"Global.epoch_num": 60, "Global.print_batch_step": 50}},
                 "cancel-{}".format(time.time()))
    print("  job_id =", jid)
    # 等它真的跑起来
    for _ in range(60):
        info = call("/api/v1/train/jobs/" + jid)
        if info["status"] == "running":
            break
        time.sleep(1)
    print("  取消前 status =", info["status"], "pid 存活中")
    t0 = time.time()
    res = call("/api/v1/train/jobs/{}/cancel".format(jid), {}, method="POST")
    print("  cancel 返回:", res)
    final = wait_terminal(jid, timeout=60)
    print("  终态: status={} exit_reason={} 收敛耗时={}s".format(
        final["status"], final["exit_reason"], round(time.time() - t0, 1)))
    ok = final["status"] == "cancelled"
    print("  结果:", "PASS" if ok else "FAIL")
    return ok


def case_setup_failure():
    print("\n== 用例2：setup 阶段失败（无 end 事件，必须判 failed 而不是卡 running）==")
    jid = submit({"model_name": "yolov8n-det",
                  "params": {"Global.epoch_num": 1,
                             "Train.dataset.data_dir": "./datasets/__不存在__",
                             "Train.dataset.label_file_list": ["./datasets/__不存在__/train.txt"]}},
                 "setupfail-{}".format(time.time()))
    print("  job_id =", jid)
    final = wait_terminal(jid, timeout=180)
    print("  终态: status={} exit_reason={} exit_code={}".format(
        final["status"], final["exit_reason"], final["exit_code"]))
    print("  error:", (final["error"] or "")[:88])
    m = call("/api/v1/train/jobs/{}/metrics?offset=-1".format(jid))
    print("  指标事件数 =", len(m["items"]), "（setup 失败应为 0，无 end）")
    ok = final["status"] == "failed" and final["exit_reason"] == "setup_failed"
    print("  结果:", "PASS" if ok else "FAIL")
    return ok


def case_queue_serial():
    print("\n== 用例3：并发闸门（max_concurrent=1，第二个必须排队）==")
    a = submit({"model_name": "yolov8n-det",
                "params": {"Global.epoch_num": 30, "Global.print_batch_step": 50}},
               "q-a-{}".format(time.time()))
    time.sleep(3)
    b = submit({"model_name": "yolov8n-det",
                "params": {"Global.epoch_num": 1, "Global.print_batch_step": 1}},
               "q-b-{}".format(time.time()))
    ia, ib = call("/api/v1/train/jobs/" + a), call("/api/v1/train/jobs/" + b)
    print("  A status={}  B status={} queue_position={}".format(
        ia["status"], ib["status"], ib["queue_position"]))
    serial = (ia["status"] == "running" and ib["status"] == "queued")
    print("  串行生效:", serial)
    call("/api/v1/train/jobs/{}/cancel".format(a), {}, method="POST")
    wait_terminal(a, timeout=60)
    call("/api/v1/train/jobs/{}/cancel".format(b), {}, method="POST")
    wait_terminal(b, timeout=60)
    print("  结果:", "PASS" if serial else "FAIL")
    return serial


def case_bad_request():
    print("\n== 用例4：非法请求在落库前就失败（不留僵尸作业）==")
    before = call("/api/v1/train/jobs?limit=500")["total"]
    try:
        call("/api/v1/train/jobs", {"model_name": "根本不存在的模型"},
             headers=dict(HDR, **{"Idempotency-Key": "bad-{}".format(time.time())}))
        print("  !! 未拒绝")
        return False
    except urllib.error.HTTPError as e:
        print("  HTTP", e.code, "->", json.loads(e.read())["detail"][:70])
    after = call("/api/v1/train/jobs?limit=500")["total"]
    print("  作业总数 {} -> {}（不应增加）".format(before, after))
    return after == before


if __name__ == "__main__":
    results = [case_cancel(), case_setup_failure(), case_queue_serial(), case_bad_request()]
    print("\n===== 汇总: {}/{} 通过 =====".format(sum(results), len(results)))
    sys.exit(0 if all(results) else 1)
