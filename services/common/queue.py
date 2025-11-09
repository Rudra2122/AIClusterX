import os, time, json, random, heapq
from typing import Optional, Dict, Any, List
import redis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

def get_redis() -> redis.Redis:
    return redis.from_url(REDIS_URL, decode_responses=True)

# Multi-queue by priority
Q_HIGH = "jobs:q:high"
Q_MED  = "jobs:q:med"
Q_LOW  = "jobs:q:low"
PRIORITY_ORDER = [Q_HIGH, Q_MED, Q_LOW]

JOB_STATUS = "jobs:status"
WORKERS_SET = "workers:set"
WORKER_QUEUE = "worker:{wid}:queue"
WORKER_HEARTBEAT = "worker:{wid}:heartbeat"
WORKER_STATS = "worker:{wid}:stats"

def _pick_queue(priority: str) -> str:
    return { "high": Q_HIGH, "med": Q_MED, "low": Q_LOW }[priority]

def enqueue_job(r: redis.Redis, job: Dict[str, Any]) -> str:
    job_id = job.get("job_id") or f"job-{int(time.time()*1000)}-{random.randint(1000,9999)}"
    job["job_id"] = job_id
    job["submit_ts"] = time.time()
    pr = job.get("priority","med")
    queue_key = _pick_queue(pr)
    # Store as JSON in list
    r.rpush(queue_key, json.dumps(job))
    r.hset(JOB_STATUS, job_id, json.dumps({"state":"queued","info":job}))
    return job_id

def get_queue_lengths(r: redis.Redis) -> Dict[str, int]:
    return {
        "high": r.llen(Q_HIGH),
        "med":  r.llen(Q_MED),
        "low":  r.llen(Q_LOW)
    }

def list_workers(r: redis.Redis) -> List[str]:
    return list(r.smembers(WORKERS_SET))

def assign_job_to_worker(r: redis.Redis, wid: str, job: Dict[str, Any]) -> None:
    r.rpush(WORKER_QUEUE.format(wid=wid), json.dumps(job))
    r.hset(JOB_STATUS, job["job_id"], json.dumps({"state":"assigned","worker":wid,"info":job}))

def pop_job_for_worker(r: redis.Redis, wid: str, timeout: int = 2) -> Optional[Dict[str, Any]]:
    # Worker consumes from its personal queue
    res = r.blpop(WORKER_QUEUE.format(wid=wid), timeout=timeout)
    if not res:
        return None
    _, payload = res
    return json.loads(payload)

def update_job_done(r: redis.Redis, job_id: str, result: Dict[str, Any], latency_sec: float, slo_violation: bool) -> None:
    r.hset(JOB_STATUS, job_id, json.dumps({
        "state":"done",
        "result":result,
        "latency_sec": latency_sec,
        "slo_violation": slo_violation
    }))

def get_job_status(r: redis.Redis, job_id: str) -> Optional[Dict[str, Any]]:
    raw = r.hget(JOB_STATUS, job_id)
    return json.loads(raw) if raw else None

def heartbeat(r: redis.Redis, wid: str, inflight: int, completed: int):
    r.sadd(WORKERS_SET, wid)
    r.set(WORKER_HEARTBEAT.format(wid=wid), int(time.time()), ex=20)
    r.hset(WORKER_STATS.format(wid=wid), mapping={"inflight": inflight, "completed": completed})
