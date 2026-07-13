FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt gunicorn whitenoise

COPY . .

ENV PYTHONPATH=/app/backend \
    HF_HOME=/hf_cache \
    TRANSFORMERS_CACHE=/hf_cache

RUN cd backend && python manage.py collectstatic --no-input

EXPOSE 8000
CMD cd backend && gunicorn repoinsight.wsgi:application
