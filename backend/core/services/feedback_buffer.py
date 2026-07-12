import os
from datetime import datetime

import redis
from django.db import models as db_models

REDIS_TTL = 86400
FLUSH_THRESHOLD = 20


def _get_redis() -> "redis.Redis":
    return redis.from_url(os.environ["REDIS_URL"])


def _fb_key(user_id, skill):
    return f"fb:{user_id}:{skill}"


def _count_key(user_id):
    return f"fb:count:{user_id}"


def increment_feedback(user_id: int, skill: str, is_up: bool):
    r = _get_redis()
    pipe = r.pipeline()
    pipe.hincrby(_fb_key(user_id, skill), "total", 1)
    if is_up:
        pipe.hincrby(_fb_key(user_id, skill), "up", 1)
    pipe.expire(_fb_key(user_id, skill), REDIS_TTL)
    pipe.incr(_count_key(user_id))
    pipe.expire(_count_key(user_id), REDIS_TTL)
    pipe.execute()


def _to_int(val: object) -> int:
    return int(val) if val else 0  # type: ignore[arg-type]


def _decode(data: object) -> dict[bytes, bytes]:
    return data if isinstance(data, dict) else {}  # type: ignore[return-value]


def get_pending_count(user_id: int) -> int:
    r = _get_redis()
    return _to_int(r.get(_count_key(user_id)))


def get_pending_scores(user_id: int, skills: set[str]) -> dict[str, dict]:
    r = _get_redis()
    result = {}
    for skill in skills:
        data = _decode(r.hgetall(_fb_key(user_id, skill)))
        if data:
            result[skill] = {
                "up": _to_int(data.get(b"up", 0)),
                "total": _to_int(data.get(b"total", 0)),
            }
    return result


def flush_user_feedback(user_id: int) -> int:
    from django.db import transaction

    from ..models import SkillFeedbackSummary

    r = _get_redis()
    cursor = 0
    keys: list[bytes] = []
    while True:
        cursor, scan_keys = r.scan(cursor=cursor, match=_fb_key(user_id, "*"))  # type: ignore[arg-type]
        keys.extend(scan_keys)
        if cursor == 0:
            break

    if not keys:
        return 0

    now = datetime.now()
    flushed = 0
    with transaction.atomic():
        for key in keys:
            raw = key.decode()
            skill = raw.split(":", 2)[2]
            data = _decode(r.hgetall(raw))
            up = _to_int(data.get(b"up", 0))
            total = _to_int(data.get(b"total", 0))
            if total == 0:
                continue

            summary, created = SkillFeedbackSummary.objects.get_or_create(
                user_id=user_id,
                skill=skill,
                defaults={"thumbs_up": up, "total": total, "last_updated": now},
            )
            if not created:
                SkillFeedbackSummary.objects.filter(
                    user_id=user_id, skill=skill
                ).update(
                    thumbs_up=db_models.F("thumbs_up") + up,
                    total=db_models.F("total") + total,
                    last_updated=now,
                )
            flushed += total

    if keys:
        r.delete(*keys)
    r.delete(_count_key(user_id))
    return flushed
