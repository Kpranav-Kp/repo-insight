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

CMD python -c "
import subprocess, sys, time
for i in range(60):
    r = subprocess.run([sys.executable, 'manage.py', 'migrate', '--noinput'])
    if r.returncode == 0:
        break
    print(f'Migration attempt {i+1} failed, retrying in 2s...')
    time.sleep(2)
else:
    print('Migrations failed. Exiting.')
    sys.exit(1)
print('Migrations done.')
subprocess.run([sys.executable, 'manage.py', 'collectstatic', '--noinput', '--clear'])
print('Starting gunicorn...')
subprocess.run(['gunicorn', '-c', 'gunicorn.conf.py', 'repoinsight.wsgi:application'])
"
