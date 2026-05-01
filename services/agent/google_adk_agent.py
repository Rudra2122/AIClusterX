"""
AIClusterX — Google Agent Development Kit (ADK) Integration.

Wraps the cluster's MCP tools as Google ADK FunctionTools and defines
a hierarchical multi-agent system:

  ClusterSupervisorAgent   (root — orchestrates the two sub-agents)
      │
      ├── JobSubmitterAgent   (submits workloads, tracks job IDs)
      └── ClusterOpsAgent     (monitors health, surfaces SLO violations)

The supervisor uses a ReAct-style loop via Google ADK's built-in runner:
  • natural-language request → supervisor reasons → delegates to sub-agent
  • sub-agent calls FunctionTool → gets result → reports back to supervisor
  • supervisor synthesises a final response

Usage:
    from services.agent.google_adk_agent import run_adk_agent
    result = run_adk_agent("Submit a high-priority torch_cnn job and report its status")
"""

import json
import os
import time
from typing import Optional

# ── Google ADK imports ────────────────────────────────────────────────────────
from google.adk.agents import LlmAgent, BaseAgent
from google.adk.tools import FunctionTool
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.models.lite_llm import LiteLlm          # thin OpenAI wrapper
from google.genai import types as genai_types

# ── Cluster tool implementations (thin wrappers around MCP server logic) ─────
# We re-use the same Redis logic rather than calling the MCP server over HTTP,
# so the agent works even during local dev without the MCP server running.

import redis as redis_lib
import random

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

def _r() -> redis_lib.Redis:
    return redis_lib.from_url(REDIS_URL, decode_responses=True)


# ── Cluster tool functions (called by ADK FunctionTools) ─────────────────────

def submit_cluster_job(
    workload: str,
    size: int = 64,
    iterations: int = 10,
    priority: str = "med",
    deadline_sec: int = 10,
) -> str:
    """
    Submit a distributed AI workload to the AIClusterX cluster.

    Args:
        workload:     Workload type: matmul | conv | sleep | torch_cnn | torch_ddp_mock
        size:         Problem / batch size (1–8192)
        iterations:   Steps or iterations (1–5000)
        priority:     Queue priority: high | med | low
        deadline_sec: SLO deadline in seconds

    Returns:
        JSON string with job_id and queue name.
    """
    r       = _r()
    job_id  = f"adk-{int(time.time()*1000)}-{random.randint(1000,9999)}"
    payload = {
        "job_id":       job_id,
        "workload":     workload,
        "size":         max(1, min(8192, size)),
        "iterations":   max(1, min(5000, iterations)),
        "priority":     priority,
        "deadline_sec": deadline_sec,
        "submit_ts":    time.time(),
        "source":       "google_adk_agent",
    }
    q_map = {"high": "jobs:q:high", "med": "jobs:q:med", "low": "jobs:q:low"}
    q_key = q_map.get(priority, "jobs:q:med")
    r.rpush(q_key, json.dumps(payload))
    r.hset("jobs:status", job_id, json.dumps({"state": "queued", "info": payload}))
    return json.dumps({"job_id": job_id, "queue": q_key})


def poll_job_status(job_id: str) -> str:
    """
    Poll the current execution status of a submitted cluster job.

    Args:
        job_id: The job ID returned by submit_cluster_job.

    Returns:
        JSON string with state, latency_sec, slo_violation, and result fields.
    """
    r   = _r()
    raw = r.hget("jobs:status", job_id)
    if not raw:
        return json.dumps({"error": f"job '{job_id}' not found"})
    return raw


def get_cluster_health() -> str:
    """
    Return current cluster health: queue depths and alive worker count.

    Returns:
        JSON string with queue lengths and worker list.
    """
    r   = _r()
    ql  = {
        "high": r.llen("jobs:q:high"),
        "med":  r.llen("jobs:q:med"),
        "low":  r.llen("jobs:q:low"),
    }
    worker_ids = list(r.smembers("workers:set") or [])
    alive = []
    now   = time.time()
    for wid in worker_ids:
        hb = r.get(f"worker:{wid}:heartbeat")
        if hb and (now - int(hb)) < 20:
            alive.append(wid)
    return json.dumps({"queues": ql, "workers": {"alive": len(alive), "ids": alive}})


def check_slo_violations(limit: int = 10) -> str:
    """
    Scan the last N completed jobs and surface any SLO violations.

    Args:
        limit: Max number of completed jobs to scan (default 10).

    Returns:
        JSON list of jobs where slo_violation == True.
    """
    r        = _r()
    all_jobs = r.hgetall("jobs:status")
    violations = []
    for job_id, raw in list(all_jobs.items())[:limit]:
        st = json.loads(raw)
        if st.get("state") == "done" and st.get("slo_violation"):
            violations.append({"job_id": job_id, **st})
    return json.dumps(violations)


# ── ADK FunctionTools ─────────────────────────────────────────────────────────

submit_tool        = FunctionTool(func=submit_cluster_job)
poll_tool          = FunctionTool(func=poll_job_status)
health_tool        = FunctionTool(func=get_cluster_health)
slo_violation_tool = FunctionTool(func=check_slo_violations)


# ── Sub-agents ────────────────────────────────────────────────────────────────

MODEL = LiteLlm(model=f"openai/gpt-4o-mini")

job_submitter_agent = LlmAgent(
    name="JobSubmitterAgent",
    model=MODEL,
    description=(
        "Handles workload submission. Accepts job specifications, submits them to the "
        "AIClusterX cluster, and returns the job ID with queue placement confirmation."
    ),
    instruction=(
        "You are a job submission specialist for a distributed AI cluster. "
        "When given a workload request, call submit_cluster_job with the correct parameters. "
        "After submitting, confirm the job_id and estimated queue wait. "
        "If the user specifies urgency, use priority='high' and a tight deadline_sec."
    ),
    tools=[submit_tool, poll_tool],
)

cluster_ops_agent = LlmAgent(
    name="ClusterOpsAgent",
    model=MODEL,
    description=(
        "Handles cluster health monitoring and SLO compliance. "
        "Reports queue depths, worker availability, and surfaces deadline violations."
    ),
    instruction=(
        "You are a cluster SRE agent. Use get_cluster_health to report current system state "
        "and check_slo_violations to surface deadline breaches. "
        "Recommend corrective actions when the cluster is unhealthy (e.g., backlog > 50 jobs, "
        "no alive workers, or SLO violation rate > 5%)."
    ),
    tools=[health_tool, slo_violation_tool],
)


# ── Root supervisor agent ─────────────────────────────────────────────────────

cluster_supervisor = LlmAgent(
    name="ClusterSupervisorAgent",
    model=MODEL,
    description="Root orchestrator for AIClusterX. Delegates to sub-agents.",
    instruction=(
        "You are the AIClusterX cluster supervisor. You have two sub-agents:\n"
        "  • JobSubmitterAgent — use for submitting and tracking individual jobs.\n"
        "  • ClusterOpsAgent   — use for health checks and SLO monitoring.\n\n"
        "For each user request:\n"
        "  1. REASON: identify which sub-agent(s) to invoke.\n"
        "  2. ACT: delegate to the appropriate sub-agent.\n"
        "  3. OBSERVE: collect the result.\n"
        "  4. REFLECT: if a job has an SLO violation, ask ClusterOpsAgent for the "
        "     overall health before responding.\n"
        "Always surface the job_id, status, and any SLO violations in your final response."
    ),
    sub_agents=[job_submitter_agent, cluster_ops_agent],
)


# ── Runner (session + execution) ──────────────────────────────────────────────

APP_NAME = "aiclusterx_adk"

def run_adk_agent(request: str, user_id: str = "default") -> str:
    """
    Run the hierarchical ADK agent system against a natural-language request.

    Args:
        request: Free-form instruction, e.g.
                 "Submit a high-priority torch_cnn job and tell me if the cluster is healthy"
        user_id: Optional user/session identifier.

    Returns:
        Agent's final response as a string.
    """
    session_service = InMemorySessionService()
    session = session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=f"session-{int(time.time())}",
    )

    runner = Runner(
        agent=cluster_supervisor,
        app_name=APP_NAME,
        session_service=session_service,
    )

    user_message = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=request)],
    )

    final_response = ""
    for event in runner.run(
        user_id=user_id,
        session_id=session.id,
        new_message=user_message,
    ):
        if event.is_final_response() and event.content:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    final_response += part.text

    return final_response or "[No response generated]"
