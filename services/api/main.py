import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST
from typing import Dict, Any
from ..common.queue import get_redis, enqueue_job, get_job_status, get_queue_lengths
from ..common.models import SubmitJob

app = FastAPI(title="AIClusterX API")

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
