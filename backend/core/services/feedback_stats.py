MIN_SAMPLES = 10
MIN_DISTINCT_USERS = 2


def get_response_type_approval(response_type: str) -> float | None:
    """
    Returns the approval rate (0.0 to 1.0) for a given response type,
    based on all users' feedback so far.

    Returns None if there isn't enough data to trust the signal yet —
    either too few total ratings, or ratings from too few distinct users
    (protects against one person's opinion dominating the signal).
    """
    from ..models import MessageFeedback

    qs = MessageFeedback.objects.filter(
        response_type=response_type, feedback__isnull=False
    )
    total = qs.count()
    distinct_users = qs.values("user").distinct().count()

    if total < MIN_SAMPLES or distinct_users < MIN_DISTINCT_USERS:
        return None

    up = qs.filter(feedback=True).count()
    return up / total


def get_adaptive_note(response_type: str) -> str:
    """
    Returns a short instruction to append to the LLM prompt, based on
    recent approval rate for this response type. Empty string if there's
    not enough data yet, or approval is in the neutral middle range.
    """
    approval = get_response_type_approval(response_type)
    if approval is None:
        return ""

    if approval < 0.5:
        return (
            "\nNote: users have recently found these responses too vague or "
            "generic — be more specific about exact file names, functions, "
            "and concrete next steps."
        )
    if approval >= 0.8:
        return (
            "\nNote: users have responded very well to this style recently — "
            "keep the same level of clarity, structure, and specificity."
        )
    return ""