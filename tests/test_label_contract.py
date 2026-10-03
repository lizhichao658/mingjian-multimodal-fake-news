from __future__ import annotations

from mingjian.labels import LABEL_FAKE, LABEL_NAME_BY_ID, LABEL_REAL


def test_eann_weibo17_label_contract_is_fixed() -> None:
    assert LABEL_REAL == 0
    assert LABEL_FAKE == 1
    assert LABEL_NAME_BY_ID == {0: "real", 1: "fake"}