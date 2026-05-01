"""
LangGraph-based Agentic Orchestrator for AIClusterX.

Implements a ReAct (Reason + Act) workflow graph that accepts natural-language
job requests, extracts structured parameters, submits to the Redis-backed cluster,
polls for completion, and self-reflects on SLO violations to decide whether to
retry with a higher priority.

Graph topology:
    START
      │
    [analyze]       ← parse intent + extract workload params via LLM
      │
    [plan]          ← validate params, pick priority/deadline heuristics
      │
    [submit]        ← enqueue job into Redis queue (calls existing queue.py)
      │
    [monitor]       ← poll job status until done or timeout
      │
    [reflect]       ← reason about result; escalate priority and retry if SLO violated
      │
    END
"""

import json
import time
import os
from typing import TypedDict, Optional, Annotated
import operator

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

import redis as redis_lib

# ── Shared Redis connection ───────────────────────────────────────────────────

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

def _get_redis() -> redis_lib.Redis:
    return redis_lib.from_url(REDIS_URL, decode_responses=True)


# ── Graph State ───────────────────────────────────────────────────────────────

class ClusterAgentState(TypedDict):
    """
    Shared state threaded through every node in the ReAct graph.
    `messages` accumulates the LLM conversation (reason trace).
    """
    messages:       Annotated[list, add_messages]   # full ReAct trace
    raw_request:    str                             # original user input
    job_params:     dict                            # extracted & validated params
    job_id:         Optional[str]                   # set after submit
    job_status:     Optional[dict]                  # latest poll result
    retry_count:    int                             # self-reflection retry counter
    final_result:   Optional[dict]                  # surfaced to caller


MAX_RETRIES   = 2
POLL_INTERVAL = 0.5   # seconds between status polls
POLL_TIMEOUT  = 30    # seconds before giving up


# ── LLM setup ────────────────────────────────────────────────────────────────

def _llm() -> ChatOpenAI:
    """Return a cheap, fast model for the ReAct reasoning steps."""
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
    )


# ── Node: analyze ─────────────────────────────────────────────────────────────

_ANALYZE_SYSTEM = """
You are an AI cluster scheduler agent. Your job is to parse a natural-language
workload request into a JSON object with exactly these fields:

{
  "workload":    <"matmul"|"conv"|"sleep"|"torch_cnn"|"torch_ddp_mock">,
  "size":        <integer 1-8192>,
  "iterations":  <integer 1-5000>,
  "priority":    <"high"|"med"|"low">,
  "deadline_sec":<integer seconds, optional>
}

Rules:
- If workload type is ambiguous, default to "torch_cnn".
- If size is not specified, default to 64.
- If iterations is not specified, default to 10.
- If priority is not specified, default to "med".
- Respond with ONLY valid JSON. No explanation, no markdown fences.
"""

def analyze_node(state: ClusterAgentState) -> ClusterAgentState:
    """
    ReAct — REASON step.
    Use the LLM to extract structured job parameters from the raw request.
    """
    llm = _llm()
    response = llm.invoke([
        SystemMessage(content=_ANALYZE_SYSTEM),
        HumanMessage(content=state["raw_request"]),
    ])
    reason_msg = AIMessage(content=f"[analyze] Extracted params: {response.content}")

    try:
        params = json.loads(response.content)
    except json.JSONDecodeError:
        # Fallback to safe defaults if LLM output is malformed
        params = {"workload": "torch_cnn", "size": 64, "iterations": 10,
                  "priority": "med", "deadline_sec": 10}

    return {
        **state,
        "messages":  [reason_msg],
        "job_params": params,
    }


# ── Node: plan ────────────────────────────────────────────────────────────────

def plan_node(state: ClusterAgentState) -> ClusterAgentState:
    """
    ReAct — REASON step.
    Apply heuristics to validate and enrich job params before submission.
    On retries, escalate priority one level (low→med→high).
    """
    params = dict(state["job_params"])
    retry  = state["retry_count"]

    # Clamp values to safe ranges
    params["size"]       = max(1,  min(8192, int(params.get("size",       64))))
    params["iterations"] = max(1,  min(5000, int(params.get("iterations", 10))))

    # Self-reflection: escalate priority on retry
    escalation = {"low": "med", "med": "high", "high": "high"}
    if retry > 0:
        old_priority      = params.get("priority", "med")
        params["priority"] = escalation[old_priority]
        reason = (f"[plan] Retry #{retry}: escalating priority "
                  f"{old_priority} → {params['priority']}")
    else:
        reason = f"[plan] Validated params: {params}"

    return {
        **state,
        "messages":  [AIMessage(content=reason)],
        "job_params": params,
    }


# ── Node: submit ──────────────────────────────────────────────────────────────

def submit_node(state: ClusterAgentState) -> ClusterAgentState:
    """
    ReAct — ACT step.
    Enqueue the job into the Redis priority queue (reuses existing queue logic).
    """
    import random

    r      = _get_redis()
    params = state["job_params"]

    # Build job payload matching existing schema
    job_id  = f"agent-{int(time.time()*1000)}-{random.randint(1000,9999)}"
    payload = {
        "job_id":       job_id,
        "workload":     params.get("workload",    "torch_cnn"),
        "size":         params.get("size",        64),
        "iterations":   params.get("iterations",  10),
        "priority":     params.get("priority",    "med"),
        "deadline_sec": params.get("deadline_sec", 10),
        "submit_ts":    time.time(),
        "source":       "langgraph_agent",     # tag so we can filter in metrics
    }

    q_map = {"high": "jobs:q:high", "med": "jobs:q:med", "low": "jobs:q:low"}
    q_key = q_map.get(payload["priority"], "jobs:q:med")

    r.rpush(q_key, json.dumps(payload))
    r.hset("jobs:status", job_id, json.dumps({"state": "queued", "info": payload}))

    return {
        **state,
        "messages": [AIMessage(content=f"[submit] Job {job_id} enqueued → {q_key}")],
        "job_id":   job_id,
    }


# ── Node: monitor ─────────────────────────────────────────────────────────────

def monitor_node(state: ClusterAgentState) -> ClusterAgentState:
    """
    ReAct — ACT + OBSERVE step.
    Poll Redis for job completion; timeout after POLL_TIMEOUT seconds.
    """
    r      = _get_redis()
    job_id = state["job_id"]
    start  = time.time()

    while time.time() - start < POLL_TIMEOUT:
        raw = r.hget("jobs:status", job_id)
        if raw:
            st = json.loads(raw)
            if st.get("state") == "done":
                return {
                    **state,
                    "messages":   [AIMessage(content=f"[monitor] Job {job_id} done: {st}")],
                    "job_status": st,
                }
        time.sleep(POLL_INTERVAL)

    # Timed out
    timeout_status = {"state": "timeout", "job_id": job_id}
    return {
        **state,
        "messages":   [AIMessage(content=f"[monitor] Job {job_id} timed out after {POLL_TIMEOUT}s")],
        "job_status": timeout_status,
    }


# ── Node: reflect ─────────────────────────────────────────────────────────────

def reflect_node(state: ClusterAgentState) -> ClusterAgentState:
    """
    ReAct — REASON step (self-reflection).
    Evaluate outcome. Decide whether to surface result or trigger a retry.
    Retry only if:
      - SLO was violated (latency > deadline)
      - retry_count < MAX_RETRIES
    """
    status = state.get("job_status", {})
    slo_violation = status.get("slo_violation", False)
    timed_out     = status.get("state") == "timeout"
    retry_count   = state["retry_count"]

    if (slo_violation or timed_out) and retry_count < MAX_RETRIES:
        reason = (f"[reflect] SLO violation detected (retry {retry_count+1}/{MAX_RETRIES}). "
                  f"Will escalate priority and resubmit.")
        return {
            **state,
            "messages":    [AIMessage(content=reason)],
            "retry_count": retry_count + 1,
            "job_id":      None,
            "job_status":  None,
        }

    # Surface final result
    result = {
        "job_id":       state.get("job_id"),
        "status":       status,
        "retry_count":  retry_count,
        "slo_met":      not slo_violation and not timed_out,
        "agent_trace":  [m.content for m in state["messages"]],
    }
    reason = f"[reflect] Final result assembled. SLO met: {result['slo_met']}"
    return {
        **state,
        "messages":     [AIMessage(content=reason)],
        "final_result": result,
    }


# ── Conditional edge: retry or done? ─────────────────────────────────────────

def _should_retry(state: ClusterAgentState) -> str:
    """Route back to plan if retrying, otherwise finish."""
    if state.get("final_result") is None:
        return "retry"
    return "done"


# ── Build the graph ───────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    g = StateGraph(ClusterAgentState)

    g.add_node("analyze", analyze_node)
    g.add_node("plan",    plan_node)
    g.add_node("submit",  submit_node)
    g.add_node("monitor", monitor_node)
    g.add_node("reflect", reflect_node)

    g.add_edge(START,     "analyze")
    g.add_edge("analyze", "plan")
    g.add_edge("plan",    "submit")
    g.add_edge("submit",  "monitor")
    g.add_edge("monitor", "reflect")

    # Self-reflection loop: reflect → plan (retry) or END
    g.add_conditional_edges(
        "reflect",
        _should_retry,
        {"retry": "plan", "done": END},
    )

    return g.compile()


# ── Public API ────────────────────────────────────────────────────────────────

def run_agent(request: str) -> dict:
    """
    Entry point. Accepts a natural-language workload request,
    runs the full ReAct graph, returns the final result dict.

    Example:
        run_agent("Run a torch CNN job, batch 32, 20 steps, high priority, 5s deadline")
    """
    graph = build_graph()
    initial_state: ClusterAgentState = {
        "messages":     [],
        "raw_request":  request,
        "job_params":   {},
        "job_id":       None,
        "job_status":   None,
        "retry_count":  0,
        "final_result": None,
    }
    final_state = graph.invoke(initial_state)
    return final_state.get("final_result", {})
