# Deploying AIClusterX on local Kubernetes (kind or minikube)

Run these on your own machine — this sandbox has no Docker/kubectl available, so
none of this has been executed for you. Everything below was written against the
actual code in this repo (Dockerfiles, ports, env vars, entrypoints), not guessed.

## 0. One real bug fixed along the way

`services/api/Dockerfile` only installed `services/api/requirements.txt`, but
`services/api/main.py` imports `services/agent/langgraph_orchestrator.py`,
`services/agent/google_adk_agent.py`, and `services/mcp_server/server.py` at
module load time. Those packages (`langgraph`, `langchain`, `google-adk`,
`litellm`, `mcp`, `fastmcp`) were never installed into the api image, so the
container would crash on startup — under `docker compose up` too, not just K8s.
Fixed by installing all three requirement files into the api image. Worth
knowing about since it means the project hadn't actually been booted
successfully as-is.

## 1. Prerequisites

```bash
# Docker Desktop (or another Docker engine) must already be running.
brew install kind kubectl        # macOS; use your package manager otherwise
kind create cluster --name aiclusterx
kubectl cluster-info --context kind-aiclusterx
```

If you'd rather use minikube: `minikube start --cpus=4 --memory=6g`.

## 2. Install metrics-server (required for the HPA)

kind/minikube don't ship this by default, and `worker-hpa` needs it to read CPU usage.

```bash
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# kind's metrics-server needs kubelet TLS verification disabled to work locally:
kubectl patch deployment metrics-server -n kube-system --type='json' \
  -p='[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'

# minikube instead: minikube addons enable metrics-server

# Verify (may take ~1 min to start reporting):
kubectl top nodes
```

## 3. Build the images

From the project root (same context docker-compose already uses):

```bash
docker build -t aiclusterx-api:latest       -f services/api/Dockerfile       .
docker build -t aiclusterx-scheduler:latest -f services/scheduler/Dockerfile .
docker build -t aiclusterx-worker:latest    -f services/worker/Dockerfile    .
```

`aiclusterx-worker` pulls in `torch==2.4.1` — the build can take several minutes
on first run.

## 4. Load the images into the cluster

Local clusters can't pull from your Docker daemon by default.

```bash
# kind
kind load docker-image aiclusterx-api:latest       --name aiclusterx
kind load docker-image aiclusterx-scheduler:latest --name aiclusterx
kind load docker-image aiclusterx-worker:latest     --name aiclusterx

# minikube (alternative)
minikube image load aiclusterx-api:latest
minikube image load aiclusterx-scheduler:latest
minikube image load aiclusterx-worker:latest
```

## 5. Set the OpenAI key secret

```bash
kubectl create namespace aiclusterx
kubectl create secret generic aiclusterx-secrets \
  --namespace aiclusterx \
  --from-literal=OPENAI_API_KEY=sk-your-real-key-or-a-placeholder
```

(A placeholder is fine unless you want to exercise `/agent/run` or `/agent/adk`.)

## 6. Apply the manifests

```bash
kubectl apply -k k8s/
```

This applies everything except the secret (already created by hand in step 5 —
kept out of `kustomization.yaml` on purpose so a real key never ends up in a
committed file).

## 7. Verify

```bash
kubectl get pods -n aiclusterx
kubectl get svc -n aiclusterx
kubectl get hpa -n aiclusterx
```

All pods should reach `Running`/`1/1 Ready` within a minute or two (the worker
image is the slowest to start, since it imports torch).

## 8. Access it

```bash
kubectl port-forward -n aiclusterx svc/api 8000:8000 &
kubectl port-forward -n aiclusterx svc/prometheus 9090:9090 &
kubectl port-forward -n aiclusterx svc/grafana 3000:3000 &
```

- API docs: http://localhost:8000/docs
- Prometheus: http://localhost:9090 — check Status → Targets, you should see the
  api/scheduler/worker pods discovered automatically
- Grafana: http://localhost:3000 (admin/admin)

```bash
curl -X POST http://localhost:8000/submit \
  -H "Content-Type: application/json" \
  -d '{"workload":"torch_cnn","size":64,"iterations":10,"priority":"high","deadline_sec":5}'
```

## 9. Demonstrate the HPA actually scaling

```bash
kubectl apply -f k8s/12-loadgen-job.yaml
kubectl get hpa worker-hpa -n aiclusterx --watch
# in another terminal:
kubectl get pods -n aiclusterx -l app=worker --watch
```

You should see `worker-hpa`'s CPU utilization climb past 50%, replicas increase
from 2 toward 8, then scale back down ~60s after load subsides. This — actual
`kubectl get hpa` output showing replica count change under load — is the
evidence worth screenshotting for a resume claim; a static deployment.yaml that
was never run doesn't demonstrate anything.

```bash
kubectl delete job loadgen -n aiclusterx   # cleanup after the demo
```

## 10. Tear down

```bash
kind delete cluster --name aiclusterx
# or: minikube delete
```
