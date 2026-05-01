import time, json
import redis
from typing import Optional, Tuple
from prometheus_client import Gauge, Counter, start_http_server

from ..common.queue import get_redis, get_queue_lengths, list_workers, assign_job_to_worker
from ..common.queue import Q_HIGH, Q_MED, Q_LOW  # raw keys

ASSIGNMENTS = Counter("aiclusterx_scheduler_assignments_total", "Jobs assigned by scheduler")
IDLE_LOOPS = Counter("aiclusterx_scheduler_idle_loops_total", "Scheduler loops with nothing to do")
QH = Gauge("aiclusterx_q_high_len", "High queue length")
QM = Gauge("aiclusterx_q_med_len", "Med queue length")
QL = Gauge("aiclusterx_q_low_len", "Low queue length")
WORKERS_ALIVE = Gauge("aiclusterx_workers_alive", "Alive workers")

LOOP_SEC = 0.6

def _best_worker(r: redis.Redis) -> Optional[str]:
    best, best_load = None, None
    alive = 0
    for wid in list_workers(r):
        stats = r.hgetall(f"worker:{wid}:stats") or {}
        hb = r.get(f"worker:{wid}:heartbeat")
        is_alive = False
        if hb:
            try:
                is_alive = (time.time() - int(hb)) < 20
            except:
                pass
        if not is_alive: 
            continue
        alive += 1
        inflight = int(stats.get("inflight", 0))
        if best_load is None or inflight < best_load:
            best, best_load = wid, inflight
    WORKERS_ALIVE.set(alive)
    return best

def _queue_pop_with_deadline(r: redis.Redis) -> Optional[Tuple[str, dict]]:
    """
    Priority order: high -> med -> low.
    Within a priority, we do FIFO pop (LPOP) but allow a tiny deadline-awareness:
    we peek up to N items to choose the earliest deadline first.
    """
    def pop_from_list(key: str) -> Optional[Tuple[str, dict]]:
        N = min(4, r.llen(key))  # small peek window
        if N == 0:
            return None
        # Peek first N payloads, choose earliest deadline
        payloads = [r.lindex(key, i) for i in range(N)]
        decoded = [json.loads(p) for p in payloads]
        # choose earliest submit_ts + deadline_sec
        best_i, best_deadline = 0, None
        now = time.time()
        for i, job in enumerate(decoded):
            dl = (job.get("submit_ts", now) + float(job.get("deadline_sec", 1e9)))
            if best_deadline is None or dl < best_deadline:
                best_deadline = dl; best_i = i
        # Remove that element by value (LREM 1 <payload>)
        picked_payload = payloads[best_i]
        r.lrem(key, 1, picked_payload)
        return key, json.loads(picked_payload)

    for key in (Q_HIGH, Q_MED, Q_LOW):
        res = pop_from_list(key)
        if res:
            return res
    return None

def main():
    start_http_server(9100)  # expose scheduler metrics
    r = get_redis()
    while True:
        q = get_queue_lengths(r)
        QH.set(q["high"]); QM.set(q["med"]); QL.set(q["low"])
        sel = _queue_pop_with_deadline(r)
        if not sel:
            IDLE_LOOPS.inc()
            time.sleep(LOOP_SEC)
            continue

        _, job = sel
        wid = _best_worker(r)
        if not wid:
            # no alive workers: push back into same priority
            pr = job.get("priority","med")
            key = { "high": Q_HIGH, "med": Q_MED, "low": Q_LOW }[pr]
            r.lpush(key, json.dumps(job))  # return to queue head
            time.sleep(1.0)
            continue

        assign_job_to_worker(r, wid, job)
        ASSIGNMENTS.inc()

if __name__ == "__main__":
    main()
