import os
from celery import Celery

# Set default configuration
# host.docker.internal allows the container to talk to the host's services if needed, but for Redis we use the service name 'redis'
# Set default configuration
# host.docker.internal allows the container to talk to the host's services if needed, but for Redis we use the service name 'redis'
# Default to localhost for running scripts on host machine. Docker will override this via env var.
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

app = Celery('tender_worker', broker=REDIS_URL, backend=REDIS_URL)

app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Asia/Kolkata',
    enable_utc=True,
    # Worker settings
    worker_concurrency=int(os.getenv('CELERY_CONCURRENCY', 1)),
    worker_prefetch_multiplier=1, # One task at a time per worker process
    task_acks_late=True, # Retry if worker crashes mid-task
)

# Import tasks so the worker knows about them
# Import tasks so the worker knows about them
import tasks
print("Celery Worker initialized! Tasks imported: ", list(app.tasks.keys()))

