from redis_client import redis_client
import time

# 并发限制（可调）
LIMITS = {
    "small": 2,
    "medium": 1,
    "large": 1
}

def choose_model(duration_sec: float):
    if duration_sec <= 120:
        return "small"
    elif duration_sec <= 600:
        return "medium"
    else:
        return "large-v3"


def acquire_slot(model: str, timeout=60):
    key = f"gpu_slot:{model}"

    for _ in range(timeout):
        current = int(redis_client.get(key) or 0)

        if current < LIMITS[model]:
            redis_client.incr(key)
            return True

        time.sleep(1)

    return False


def release_slot(model: str):
    key = f"gpu_slot:{model}"
    redis_client.decr(key)