FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu --no-deps \
    && pip install -r requirements.txt gunicorn whitenoise

COPY . .

WORKDIR /app/backend

ENV PYTHONPATH=/app/backend \
    HF_HOME=/hf_cache \
    TRANSFORMERS_CACHE=/hf_cache

ARG SECRET_KEY=dummy-build-key
ENV SECRET_KEY=$SECRET_KEY
RUN python manage.py collectstatic --no-input

EXPOSE 8000

CMD python entrypoint.py
