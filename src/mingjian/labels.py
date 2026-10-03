"""Canonical binary labels for Weibo17/EANN compatibility.

The official EANN preprocessing uses ``0`` for real/non-rumor posts and
``1`` for fake/rumor posts. Keep this contract centralized so data adapters,
models, metrics, and reports cannot silently flip the class direction.
"""

LABEL_REAL = 0
LABEL_FAKE = 1
LABEL_NAME_BY_ID = {LABEL_REAL: "real", LABEL_FAKE: "fake"}

__all__ = ["LABEL_FAKE", "LABEL_NAME_BY_ID", "LABEL_REAL"]