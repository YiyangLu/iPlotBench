# iPlotBench Base Image
# Minimal Python environment for agent evaluation sandbox
#
# Usage:
#   docker build -t iplotbench:latest -f docker/Dockerfile .
#
# Extend in your agent:
#   FROM iplotbench:latest
#   RUN pip install your-dependencies
#   COPY . /app/agent
#   ENTRYPOINT ["python", "-m", "your_runner"]

FROM python:3.10-slim

WORKDIR /workspace

# Create standard directories
RUN mkdir -p /data/envs /data/outputs /app/logs

# No dependencies installed - user extends and installs their own
