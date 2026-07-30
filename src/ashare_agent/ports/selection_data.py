from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol

import pandas as pd


@dataclass(frozen=True)
class SelectionDataset:
    frame: pd.DataFrame
    provider: str
    market: str
    data_cutoff: date
    provenance: dict[str, Any] = field(default_factory=dict)
    retrieval: dict[str, Any] = field(default_factory=dict)


class SelectionDataProvider(Protocol):
    name: str
    market: str

    def load(
        self,
        start: date,
        end: date,
        *,
        force_refresh: bool = False,
    ) -> SelectionDataset:
        """Return canonical, once-normalized cross-sectional market data."""
