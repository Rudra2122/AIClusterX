FROM python:3.11-slim
WORKDIR /app

# deps
COPY services/api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# source
COPY services /app/services

# run as a package
CMD ["python", "-m", "services.api.main"]
