#!/bin/sh

echo "=== ENTRYPOINT STARTED ==="
echo "Workdir: $(pwd)"
python -c "import sys; print(f'Python {sys.version}')"
echo "DB host: ${PGHOST:-not set}"

echo "--- Running migrations ---"
i=0
while [ $i -lt 60 ]; do
  if python manage.py migrate --noinput 2>&1; then
    echo "Migrations succeeded."
    break
  fi
  i=$(( i + 1 ))
  if [ $i -ge 60 ]; then
    echo "Migrations failed after $i attempts. Exiting."
    exit 1
  fi
  echo "Migration attempt $i failed, retrying in 2s..."
  sleep 2
done

echo "--- Collectstatic ---"
python manage.py collectstatic --noinput --clear || echo "Collectstatic had warnings"

echo "=== Starting gunicorn ==="
exec "$@"
