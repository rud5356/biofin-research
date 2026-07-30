FROM python:3.11-slim

WORKDIR /workspace

COPY llm/requirements.txt /tmp/llm-requirements.txt
RUN pip install --no-cache-dir -r /tmp/llm-requirements.txt

ENTRYPOINT ["python", "-u"]
