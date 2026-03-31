from celery import Celery
import os

celery = Celery(
    "whisper",
    broker=os.getenv("CELERY_BROKER_URL"),
    backend=os.getenv("REDIS_URL")
)

# 👇 不用 autodiscover，直接强制导入（最稳）
import tasks