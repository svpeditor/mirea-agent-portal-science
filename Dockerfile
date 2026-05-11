FROM python:3.12-slim

WORKDIR /agent

# Local-dev Dockerfile. Портал генерирует свой Dockerfile из manifest.yaml.

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY agent.py manifest.yaml ./

ENTRYPOINT ["python", "agent.py"]
