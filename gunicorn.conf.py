import os

project_root = os.path.dirname(os.path.abspath(__file__))
pythonpath = os.path.join(project_root, "backend")

bind = "0.0.0.0:8000"
workers = 2
timeout = 120
worker_tmp_dir = "/dev/shm"
