from __future__ import annotations

import pandas as pd

from ashare_agent.selection.boards import classify_cn_board


def test_cn_board_classification_uses_a_share_ranges_and_master_data() -> None:
    frame = pd.DataFrame(
        {
            "code": [
                "600519",
                "002594",
                "300750",
                "301001",
                "688001",
                "689009",
                "920001",
                "200002",
                "700001",
            ],
            "market": [
                "主板",
                "中小板",
                "创业板",
                pd.NA,
                "科创板",
                pd.NA,
                "北交所",
                "主板",
                "主板",
            ],
        }
    )

    assert classify_cn_board(frame).tolist() == [
        "main",
        "main",
        "chinext",
        "chinext",
        "star",
        "star",
        "unknown",
        "unknown",
        "main",
    ]
