import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST
from typing import Dict, Any
from pydantic import BaseModel

from ..common.queue import get_redis, enqueue_job, get_job_status, get_queue_lengths
from ..common.models import SubmitJob
from ..agent.langgraph_orchestrator import run_agent as run_langgraph_agent
from ..agent.google_adk_agent import run_adk_agent
from ..mcp_server.server import mcp

app = FastAPI(
    title="AIClusterX API",
    description=(
        "Distributed AI cluster with LangGraph ReAct orchestration, "
        "Google ADK multi-agent system, and MCP server integration."
    ),
)

# Mount MCP server at /mcp — compatible with Claude Desktop and any MCP client
app.mount("/mcp", mcp.streamable_http_app())

JOBS_SUBMITTED = Counter("aiclusterx_jobs_submitted_total", "Total jobs submitted")
Q_HIGH = Gauge("aiclusterx_q_high_len", "High-priority queue length")
Q_MED  = Gauge("aiclusterx_q_med_len", "Med-priority queue length")
Q_LOW  = Gauge("aiclusterx_q_low_len", "Low-priority queue length")

SLO_DEADLINE_SEC = int(os.getenv("SLO_DEADLINE_SEC", "10"))

@app.get("/healthz")
def health():
    return {"ok": True}

@app.post("/submit")
def submit(job: SubmitJob) -> Dict[str, Any]:
    r = get_redis()
    # default deadline if not provided
    if job.deadline_sec is None:
        job.deadline_sec = SLO_DEADLINE_SEC
    job_id = enqueue_job(r, job.model_dump())
    JOBS_SUBMITTED.inc()
    _update_queue_metrics(r)
    return {"job_id": job_id}

@app.get("/status/{job_id}")
def status(job_id: str):
    r = get_redis()
    st = get_job_status(r, job_id)
    if not st:
        raise HTTPException(status_code=404, detail="not found")
    _update_queue_metrics(r)
    return st

# ── Agent request schemas ─────────────────────────────────────────────────────

class AgentRequest(BaseModel):
    request: str   # natural-language workload instruction

class ADKRequest(BaseModel):
    request: str
    user_id: str = "default"


# ── LangGraph agent endpoint ──────────────────────────────────────────────────

@app.post("/agent/run")
def agent_run(body: AgentRequest) -> Dict[str, Any]:
    """
    Submit a natural-language workload request to the LangGraph ReAct agent.

    The agent: classifies intent → extracts params → submits to cluster
    → monitors completion → self-reflects on SLO violations → retries if needed.

    Example body:
        {"request": "Run a high-priority torch_cnn job, batch 32, 10 steps, 5s deadline"}
    """
    try:
        result = run_langgraph_agent(body.request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Google ADK multi-agent endpoint ──────────────────────────────────────────

@app.post("/agent/adk")
def agent_adk(body: ADKRequest) -> Dict[str, Any]:
    """
    Submit a request to the Google ADK hierarchical multi-agent system.

    Routes between JobSubmitterAgent and ClusterOpsAgent via a supervisor,
    using Google ADK's built-in ReAct runner.

    Example body:
        {"request": "Submit a torch_ddp_mock job and check if the cluster is healthy"}
    """
    try:
        response = run_adk_agent(body.request, user_id=body.user_id)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST)

def _update_queue_metrics(r):
    q = get_queue_lengths(r)
    Q_HIGH.set(q["high"])
    Q_MED.set(q["med"])
    Q_LOW.set(q["low"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
