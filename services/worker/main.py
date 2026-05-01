import os, time
import numpy as np
from typing import Tuple
from prometheus_client import Gauge, Counter, Histogram, start_http_server

import torch
import torch.nn as nn
import torch.optim as optim

from ..common.queue import get_redis, pop_job_for_worker, update_job_done, heartbeat

WID = os.getenv("WORKER_ID", "worker-unknown")
LOOP_SLEEP = 0.25

INFLIGHT = Gauge("aiclusterx_worker_inflight", "Jobs in flight", ["worker"])
COMPLETED = Counter("aiclusterx_worker_completed_total", "Jobs completed", ["worker"])
UTIL = Gauge("aiclusterx_worker_utilization", "Estimated utilization (0..1)", ["worker"])
POWER_W = Gauge("aiclusterx_worker_power_watts", "Estimated power draw (W)", ["worker"])
COST = Counter("aiclusterx_worker_cost_usd_total", "Estimated cumulative cost (USD)", ["worker"])

# New metrics:
LATENCY = Histogram("aiclusterx_job_latency_seconds", "End-to-end job latency (sec)",
                    buckets=(0.5,1,2,3,5,8,13,21,34,55))
SLO_VIOL = Counter("aiclusterx_slo_violations_total", "Jobs exceeding deadline SLO")

CPU_TDP_W = 25.0  # pseudo power model
IDLE_W = 6.0

def estimate_power(util: float) -> float:
    dyn = CPU_TDP_W - IDLE_W
    return IDLE_W + dyn * (util ** 1.4)

def do_numpy_matmul(size: int, iterations: int) -> float:
    A = np.random.rand(size, size).astype(np.float32)
    B = np.random.rand(size, size).astype(np.float32)
    start = time.time()
    for _ in range(iterations):
        _ = A @ B
    return time.time() - start

def do_numpy_conv(size: int, iterations: int) -> float:
    H = W = size; K = 3
    img = np.random.rand(H, W).astype(np.float32)
    kernel = np.random.rand(K, K).astype(np.float32)
    start = time.time()
    for _ in range(iterations):
        out = np.zeros_like(img)
        for i in range(1, H-1):
            for j in range(1, W-1):
                out[i,j] = float(np.sum(img[i-1:i+2, j-1:j+2] * kernel))
        img = out
    return time.time() - start

def do_torch_cnn(batch: int, steps: int) -> float:
    """
    Simple CPU-only toy CNN training loop (works on Mac).
    """
    device = torch.device("cpu")
    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, 3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(16*32*32, 10)
    ).to(device)
    opt = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    start = time.time()
    for _ in range(steps):
        x = torch.randn(batch, 3, 32, 32, device=device)
        y = torch.randint(0, 10, (batch,), device=device)
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
    return time.time() - start

def do_torch_ddp_mock(batch: int, steps: int) -> float:
    """
    Single-process 'DDP-style' loop that mimics gradient sync cost.
    On real multi-proc DDP you would use torchrun and multiple ranks.
    """
    device = torch.device("cpu")
    model = nn.Sequential(
        nn.Linear(2048, 2048),
        nn.ReLU(),
        nn.Linear(2048, 2048),
        nn.ReLU(),
        nn.Linear(2048, 10)
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    start = time.time()
    for step in range(steps):
        x = torch.randn(batch, 2048, device=device)
        y = torch.randint(0, 10, (batch,), device=device)
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        # Mimic comm/sync overhead (sleep tiny)
        time.sleep(0.002)
        opt.step()
    return time.time() - start

def run_workload(workload: str, size: int, iterations: int) -> float:
    if workload == "sleep":
        t0 = time.time(); time.sleep(size/1000.0); return time.time()-t0
    if workload == "matmul":
        return do_numpy_matmul(size, iterations)
    if workload == "conv":
        return do_numpy_conv(size, iterations)
    if workload == "torch_cnn":
        return do_torch_cnn(size, iterations)  # size treated as batch
    if workload == "torch_ddp_mock":
        return do_torch_ddp_mock(size, iterations)
    # default
    t0 = time.time(); time.sleep(0.05); return time.time()-t0

def main():
    # metrics on :9200
    start_http_server(9200)
    r = get_redis()
    inflight = 0
    completed = 0

    while True:
        UTIL.labels(WID).set(0.0)
        POWER_W.labels(WID).set(estimate_power(0.0))
        INFLIGHT.labels(WID).set(inflight)
        heartbeat(r, WID, inflight, completed)

        job = pop_job_for_worker(r, WID, timeout=2)
        if not job:
            time.sleep(LOOP_SLEEP)
            continue

        inflight += 1
        INFLIGHT.labels(WID).set(inflight)
        UTIL.labels(WID).set(0.9)
        POWER_W.labels(WID).set(estimate_power(0.9))

        start = time.time()
        elapsed = run_workload(job.get("workload","sleep"), int(job.get("size",256)), int(job.get("iterations",1)))
        latency = time.time() - start
        LATENCY.observe(latency)

        # simple cost model
        cost = 0.00006 * latency
        COST.labels(WID).inc(cost)
        # SLO
        deadline = float(job.get("deadline_sec", 1e9))
        slo_violation = (latency > deadline)
        if slo_violation:
            SLO_VIOL.inc()

        completed += 1
        COMPLETED.labels(WID).inc()
        update_job_done(r, job["job_id"], {"elapsed_sec": elapsed, "cost_usd": cost}, latency, slo_violation)

        inflight -= 1
        INFLIGHT.labels(WID).set(inflight)
        UTIL.labels(WID).set(0.1)
        POWER_W.labels(WID).set(estimate_power(0.1))
        time.sleep(0.05)

if __name__ == "__main__":
    main()
