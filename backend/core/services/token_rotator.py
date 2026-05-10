import threading


class TokenRotator:
    def __init__(self, tokens: list[str]):
        self._tokens = tokens
        self._index = 0
        self._lock = threading.Lock()

    def next(self) -> str:
        with self._lock:
            token = self._tokens[self._index]
            self._index = (self._index + 1) % len(self._tokens)
        return token
