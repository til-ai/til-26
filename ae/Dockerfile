# Dockerfile for building the AE image.


# The base image. Using a regular Python base because AE agents don't typically
# require GPU. This will speed up your evaluation.
FROM python:3.11-slim

# Configures settings for the image.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
WORKDIR /workspace

# Installs your dependencies.
RUN pip install -U pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copies your source files.
COPY src .

# Starts your model server.
CMD uvicorn ae_server:app --port 5005 --host 0.0.0.0
