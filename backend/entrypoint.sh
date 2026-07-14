#!/bin/sh
set -e

echo "Waiting for database..."
for i in $(seq 1 30); do
  python manage.py migrate --noinput 2>&1 && break
  echo "Migration attempt $i failed, retrying in 2s..."
  sleep 2
done

python manage.py collectstatic --noinput --clear

exec "$@"
