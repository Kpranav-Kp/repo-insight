#!/bin/sh
set -e

python backend/manage.py migrate --noinput
python backend/manage.py collectstatic --noinput --clear

exec "$@"
