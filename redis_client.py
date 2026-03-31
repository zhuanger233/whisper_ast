import redis
import os
import json

redis_client = redis.Redis.from_url(
    os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    decode_responses=True
)

def set_job(job_id: str, data: dict, merge: bool = True):
    if merge:
        existing = get_job(job_id) or {}
        data = {**existing, **data}
    redis_client.set(f"job:{job_id}", json.dumps(data))


def get_job(job_id: str):
    data = redis_client.get(f"job:{job_id}")
    return json.loads(data) if data else None
