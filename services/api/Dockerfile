FROM python:3.11-slim
WORKDIR /app

# The API imports the agent + mcp_server modules directly (see main.py:
# `from ..agent.langgraph_orchestrator import ...`, `from ..mcp_server.server import mcp`),
# so all three requirement sets have to land in this image, not just api/requirements.txt.
COPY services/api/requirements.txt /app/req-api.txt
COPY services/agent/requirements.txt /app/req-agent.txt
COPY services/mcp_server/requirements.txt /app/req-mcp.txt
RUN pip install --no-cache-dir -r /app/req-api.txt -r /app/req-agent.txt -r /app/req-mcp.txt

# source
COPY services /app/services

# run as a package
CMD ["python", "-m", "services.api.main"]
