# demo.sh
#!/bin/bash
set -e

echo "🚀 AIClusterX Demo Starting..."

# 1️⃣ Submit a mix of jobs
echo "Submitting mixed workloads..."
for i in {1..10}; do
  curl -s -X POST http://localhost:8000/submit -H "Content-Type: application/json" \
  -d '{"workload":"torch_ddp_mock","size":128,"iterations":20,"priority":"med"}' >/dev/null
done

for i in {1..5}; do
  curl -s -X POST http://localhost:8000/submit -H "Content-Type: application/json" \
  -d '{"workload":"matmul","size":256,"iterations":8,"priority":"low","deadline_sec":12}' >/dev/null
done

for i in {1..3}; do
  curl -s -X POST http://localhost:8000/submit -H "Content-Type: application/json" \
  -d '{"workload":"torch_cnn","size":64,"iterations":12,"priority":"high","deadline_sec":6}' >/dev/null
done
echo "✅ Workloads submitted."

# 2️⃣ Monitor queues
echo "Watching Prometheus metrics for queue length and assignments..."
echo "Open: http://localhost:9090  (Prometheus)"
echo "      http://localhost:3000  (Grafana - admin/admin)"
sleep 5

# 3️⃣ Optional fault tolerance test
read -p "Do you want to test fault tolerance? (y/n): " resp
if [[ "$resp" == "y" ]]; then
  worker=$(docker ps --format '{{.Names}}' | grep worker1 | head -n 1)
  echo "Stopping $worker temporarily..."
  docker stop "$worker"
  sleep 5
  echo "Restarting worker..."
  docker start "$worker"
fi

echo "✅ Demo Complete. Capture screenshots of Prometheus/Grafana dashboards!"
