from __future__ import annotations

from typing import Literal, cast

import pandas as pd


BoardScope = Literal["main", "main_chinext", "main_chinext_star"]

BOARD_SCOPES: tuple[BoardScope, ...] = (
    "main",
    "main_chinext",
    "main_chinext_star",
)

_BOARDS_BY_SCOPE: dict[BoardScope, frozenset[str]] = {
    "main": frozenset({"main"}),
    "main_chinext": frozenset({"main", "chinext"}),
    "main_chinext_star": frozenset({"main", "chinext", "star"}),
}


def normalize_board_scope(value: str) -> BoardScope:
    scope = value.strip().lower()
    if scope not in BOARD_SCOPES:
        supported = ", ".join(BOARD_SCOPES)
        raise ValueError(
            f"board_scope must be one of: {supported}; got {value!r}"
        )
    return cast(BoardScope, scope)


def boards_for_scope(value: str) -> frozenset[str]:
    return _BOARDS_BY_SCOPE[normalize_board_scope(value)]


def classify_cn_board(frame: pd.DataFrame) -> pd.Series:
    """Classify A-share rows using master data, with a code-range fallback."""

    master_board = pd.Series("unknown", index=frame.index, dtype="string")
    if "market" in frame.columns:
        market = frame["market"].astype("string").str.strip().str.lower()
        master_board.loc[
            market.str.contains("科创", na=False)
            | market.isin({"star", "star market", "sci-tech innovation board"})
        ] = "star"
        master_board.loc[
            market.str.contains("创业", na=False)
            | market.isin({"chinext", "gem"})
        ] = "chinext"
        master_board.loc[
            market.str.contains("主板|中小板", regex=True, na=False)
            | market.isin({"main", "main board", "sme"})
        ] = "main"

    codes = (
        frame["code"]
        .astype("string")
        .str.strip()
        .str.upper()
        .str.split(".", n=1, regex=False)
        .str[0]
        .str.zfill(6)
    )
    board = pd.Series("unknown", index=frame.index, dtype="string")
    board.loc[
        codes.str.fullmatch(r"(?:00[0-4]|60[0135])\d{3}", na=False)
    ] = "main"
    board.loc[codes.str.fullmatch(r"30\d{4}", na=False)] = "chinext"
    board.loc[codes.str.fullmatch(r"68[89]\d{3}", na=False)] = "star"

    # Master data covers future A-share code ranges, while known B-share and
    # Beijing-exchange ranges remain outside these three scopes.
    fallback = board.eq("unknown") & ~codes.str.fullmatch(
        r"[2489]\d{5}", na=False
    )
    board.loc[fallback] = master_board.loc[fallback]
    return board
