FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime


WORKDIR /workspace/code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONUNBUFFERED=1

CMD ["bash"]
