#!/bin/sh
set -e

echo "Waiting for database..."
i=0
while [ $i -lt 60 ]; do
  python manage.py migrate --noinput 2>&1 && break
  i=$(( i + 1 ))
  if [ $i -ge 60 ]; then
    echo "Migrations failed after $i attempts. Exiting."
    exit 1
  fi
  echo "Migration attempt $i failed, retrying in 2s..."
  sleep 2
done

python manage.py collectstatic --noinput --clear

exec "$@"
