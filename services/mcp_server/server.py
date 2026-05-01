"""
AIClusterX MCP Server.

Exposes the cluster's capabilities as Model Context Protocol (MCP) tools,
allowing any MCP-compatible AI agent (Claude, Google ADK, etc.) to submit
and monitor distributed AI workloads without direct Redis or API access.

Tools exposed:
  • submit_job          — enqueue a workload into the cluster
  • get_job_status      — poll a job by ID
  • get_cluster_metrics — queue depths + alive worker count
  • list_workload_types — enumerate supported workload profiles

Run standalone:
    python -m services.mcp_server.server

Or mount on the existing FastAPI app (see services/api/main.py).
"""

import json
import os
import time
import random
from typing import Optional

import redis as redis_lib
from mcp.server.fastmcp import FastMCP

# ── Redis helpers ─────────────────────────────────────────────────────────────

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

def _r() -> redis_lib.Redis:
    return redis_lib.from_url(REDIS_URL, decode_responses=True)


# ── MCP server instance ───────────────────────────────────────────────────────

mcp = FastMCP(
    name="aiclusterx",
    instructions=(
        "AIClusterX cluster tools. Use submit_job to run a distributed AI workload, "
        "get_job_status to poll progress, get_cluster_metrics to inspect queue health, "
        "and list_workload_types to see what workloads are supported."
    ),
)


# ── Tool: submit_job ──────────────────────────────────────────────────────────

@mcp.tool()
def submit_job(
    workload: str,
    size: int = 64,
    iterations: int = 10,
    priority: str = "med",
    deadline_sec: Optional[int] = 10,
) -> dict:
    """
    Submit a distributed AI workload to the cluster.

    Args:
        workload:     Workload type. One of: matmul, conv, sleep, torch_cnn, torch_ddp_mock.
        size:         Problem size or batch size (1–8192).
        iterations:   Training steps or loop iterations (1–5000).
        priority:     Queue priority: "high", "med", or "low".
        deadline_sec: SLO deadline in seconds. Jobs exceeding this are flagged.

    Returns:
        {"job_id": str, "queue": str, "submit_ts": float}
    """
    valid_workloads = {"matmul", "conv", "sleep", "torch_cnn", "torch_ddp_mock"}
    if workload not in valid_workloads:
        return {"error": f"Unknown workload '{workload}'. Choose from: {valid_workloads}"}
    if priority not in {"high", "med", "low"}:
        return {"error": "priority must be 'high', 'med', or 'low'"}

    r = _r()
    job_id  = f"mcp-{int(time.time()*1000)}-{random.randint(1000,9999)}"
    payload = {
        "job_id":       job_id,
        "workload":     workload,
        "size":         max(1, min(8192, size)),
        "iterations":   max(1, min(5000, iterations)),
        "priority":     priority,
        "deadline_sec": deadline_sec,
        "submit_ts":    time.time(),
        "source":       "mcp_server",
    }

    q_map = {"high": "jobs:q:high", "med": "jobs:q:med", "low": "jobs:q:low"}
    q_key = q_map[priority]

    r.rpush(q_key, json.dumps(payload))
    r.hset("jobs:status", job_id, json.dumps({"state": "queued", "info": payload}))

    return {"job_id": job_id, "queue": q_key, "submit_ts": payload["submit_ts"]}


# ── Tool: get_job_status ──────────────────────────────────────────────────────

@mcp.tool()
def get_job_status(job_id: str) -> dict:
    """
    Poll the current status of a submitted job.

    Args:
        job_id: The job ID returned by submit_job.

    Returns:
        Job status dict with fields: state, worker, result, latency_sec, slo_violation.
        state is one of: queued, assigned, done.
    """
    r = _r()
    raw = r.hget("jobs:status", job_id)
    if not raw:
        return {"error": f"Job '{job_id}' not found"}
    return json.loads(raw)


# ── Tool: get_cluster_metrics ─────────────────────────────────────────────────

@mcp.tool()
def get_cluster_metrics() -> dict:
    """
    Return a snapshot of current cluster health.

    Returns:
        {
          "queues":  {"high": int, "med": int, "low": int},  # pending job counts
          "workers": {"alive": int, "ids": [str]}            # heartbeat-confirmed workers
        }
    """
    r = _r()

    queue_lengths = {
        "high": r.llen("jobs:q:high"),
        "med":  r.llen("jobs:q:med"),
        "low":  r.llen("jobs:q:low"),
    }

    worker_ids = list(r.smembers("workers:set") or [])
    alive = []
    now   = time.time()
    for wid in worker_ids:
        hb = r.get(f"worker:{wid}:heartbeat")
        if hb:
            try:
                if (now - int(hb)) < 20:
                    alive.append(wid)
            except ValueError:
                pass

    return {
        "queues":  queue_lengths,
        "workers": {"alive": len(alive), "ids": alive},
    }


# ── Tool: list_workload_types ─────────────────────────────────────────────────

@mcp.tool()
def list_workload_types() -> dict:
    """
    List all supported workload profiles with recommended parameter ranges.

    Returns:
        Dict mapping workload name → description + recommended params.
    """
    return {
        "matmul": {
            "description": "NumPy matrix multiplication. CPU-bound, linear with size^2.",
            "recommended": {"size": "128–512", "iterations": "5–50"},
        },
        "conv": {
            "description": "2D convolution loop. Memory-intensive, slow for large sizes.",
            "recommended": {"size": "32–128", "iterations": "1–5"},
        },
        "sleep": {
            "description": "Synthetic sleep workload. size/1000 seconds per job. Good for SLO tests.",
            "recommended": {"size": "100–5000", "iterations": "1"},
        },
        "torch_cnn": {
            "description": "Toy CNN training loop (3→8→16 ch, 32×32 input). CPU-only.",
            "recommended": {"size": "16–128", "iterations": "5–50"},
        },
        "torch_ddp_mock": {
            "description": "Simulated multi-node DDP with comm overhead (2ms/step sleep).",
            "recommended": {"size": "64–256", "iterations": "5–20"},
        },
    }


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    # Run as a standalone stdio MCP server (compatible with Claude Desktop, etc.)
    mcp.run(transport="stdio")
