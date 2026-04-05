import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'repoinsight.settings')
os.environ.setdefault('HF_HOME', 'C:\\HFCache')

app = Celery('repoinsight')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()