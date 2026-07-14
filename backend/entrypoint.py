#!/usr/bin/env python3
import subprocess
import sys
import time

print("=== Running migrations ===")
for i in range(60):
    r = subprocess.run([sys.executable, "manage.py", "migrate", "--noinput"])  # noqa: S603
    if r.returncode == 0:
        break
    print(f"Migration attempt {i + 1} failed, retrying in 2s...")
    time.sleep(2)
else:
    print("Migrations failed. Exiting.")
    sys.exit(1)

print("=== Collectstatic ===")
subprocess.run([sys.executable, "manage.py", "collectstatic", "--noinput", "--clear"])  # noqa: S603

print("=== Starting gunicorn ===")
subprocess.run(  # noqa: S603
    [
        sys.executable,
        "-m",
        "gunicorn",
        "-c",
        "gunicorn.conf.py",
        "repoinsight.wsgi:application",
    ]
)
