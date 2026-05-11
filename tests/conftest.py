"""Stub portal_sdk до импорта agent.py — см. proverka/tests/conftest.py."""
from __future__ import annotations

import sys
from types import SimpleNamespace


class _StubAgent:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.params: dict = {}
        self.output_dir = None

    def log(self, *_a: object, **_k: object) -> None:
        pass

    def progress(self, *_a: object, **_k: object) -> None:
        pass

    def item_done(self, *_a: object, **_k: object) -> None:
        pass

    def failed(self, *_a: object, **_k: object) -> None:
        pass

    def result(self, *_a: object, **_k: object) -> None:
        pass


sys.modules.setdefault("portal_sdk", SimpleNamespace(Agent=_StubAgent))
