from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, order=True)
class InstrumentId:
    """Provider-independent security identity.

    The local code is intentionally kept as text so leading zeroes (for
    example ``00700`` in Hong Kong) are never lost.
    """

    mic: str
    local_code: str

    def __post_init__(self) -> None:
        mic = self.mic.strip().upper()
        local_code = self.local_code.strip().upper()
        if not re.fullmatch(r"[A-Z0-9]{4}", mic):
            raise ValueError(f"Invalid MIC: {self.mic!r}")
        if not local_code or not re.fullmatch(r"[A-Z0-9._-]+", local_code):
            raise ValueError(f"Invalid local instrument code: {self.local_code!r}")
        object.__setattr__(self, "mic", mic)
        object.__setattr__(self, "local_code", local_code)

    @property
    def canonical(self) -> str:
        return f"{self.mic}:{self.local_code}"

    @classmethod
    def parse(cls, value: str) -> "InstrumentId":
        mic, separator, local_code = value.strip().upper().partition(":")
        if not separator:
            raise ValueError("Canonical instrument id must look like XSHG:600519")
        return cls(mic=mic, local_code=local_code)


@dataclass(frozen=True)
class MarketProfile:
    market_id: str
    timezone: str
    currency: str
    mic_to_suffix: dict[str, str]
    local_code_pattern: re.Pattern[str]
    default_board_lot: int | None

    @property
    def suffix_to_mic(self) -> dict[str, str]:
        return {suffix: mic for mic, suffix in self.mic_to_suffix.items()}

    def normalize_provider_symbol(self, value: str) -> str:
        raw = value.strip().upper()
        if ":" in raw:
            instrument = InstrumentId.parse(raw)
        else:
            local_code, separator, suffix = raw.rpartition(".")
            if not separator or suffix not in self.suffix_to_mic:
                examples = ", ".join(
                    f"<code>.{item}" for item in self.suffix_to_mic
                )
                raise ValueError(
                    f"Symbol {value!r} is not valid for {self.market_id}; "
                    f"expected a provider symbol such as {examples}"
                )
            instrument = InstrumentId(
                mic=self.suffix_to_mic[suffix],
                local_code=local_code,
            )
        if instrument.mic not in self.mic_to_suffix:
            raise ValueError(
                f"MIC {instrument.mic!r} is not part of market {self.market_id}"
            )
        if not self.local_code_pattern.fullmatch(instrument.local_code):
            raise ValueError(
                f"Code {instrument.local_code!r} is not valid for {self.market_id}"
            )
        return self.provider_symbol(instrument)

    def provider_symbol(self, instrument: InstrumentId) -> str:
        suffix = self.mic_to_suffix.get(instrument.mic)
        if suffix is None:
            raise ValueError(
                f"MIC {instrument.mic!r} is not part of market {self.market_id}"
            )
        if not self.local_code_pattern.fullmatch(instrument.local_code):
            raise ValueError(
                f"Code {instrument.local_code!r} is not valid for {self.market_id}"
            )
        return f"{instrument.local_code}.{suffix}"

    def instrument_from_parts(
        self,
        local_code: str,
        *,
        exchange: str | None = None,
        provider_symbol: str | None = None,
    ) -> InstrumentId:
        code = str(local_code).strip().upper()
        exchange_name = str(exchange or "").strip().upper()
        exchange_aliases = {
            "SSE": "XSHG",
            "SH": "XSHG",
            "SZSE": "XSHE",
            "SZ": "XSHE",
            "BSE": "XBSE",
            "BJ": "XBSE",
            "HKEX": "XHKG",
            "HK": "XHKG",
        }
        exchange_mic = exchange_aliases.get(exchange_name, exchange_name)

        if provider_symbol:
            normalized = self.normalize_provider_symbol(provider_symbol)
            symbol_code, suffix = normalized.rsplit(".", 1)
            instrument = InstrumentId(self.suffix_to_mic[suffix], symbol_code)
            if code != instrument.local_code:
                raise ValueError(
                    f"Local code {code!r} does not match provider symbol "
                    f"{normalized!r}"
                )
            if exchange_mic and exchange_mic != instrument.mic:
                raise ValueError(
                    f"Exchange {exchange_name!r} does not match provider symbol "
                    f"{normalized!r}"
                )
            return instrument

        mic = exchange_mic
        if not mic:
            if self.market_id == "CN":
                mic = (
                    "XBSE"
                    if code.startswith(("4", "8", "920"))
                    else "XSHG"
                    if code.startswith(("5", "6", "9"))
                    else "XSHE"
                )
            elif self.market_id == "HK":
                mic = "XHKG"
        instrument = InstrumentId(mic=mic, local_code=code)
        self.provider_symbol(instrument)
        return instrument

    def board_lot(self, instrument: InstrumentId, metadata: dict[str, Any]) -> int:
        del instrument
        configured = metadata.get("board_lot")
        if configured is not None:
            lot = int(configured)
            if lot < 1:
                raise ValueError("board_lot must be positive")
            return lot
        if self.default_board_lot is None:
            raise ValueError(
                f"{self.market_id} board lot must come from instrument master data"
            )
        return self.default_board_lot


_MARKETS: dict[str, MarketProfile] = {}


def register_market(profile: MarketProfile, *, replace: bool = False) -> None:
    key = profile.market_id.strip().upper()
    if key in _MARKETS and not replace:
        raise ValueError(f"Market profile {key!r} is already registered")
    _MARKETS[key] = profile


def get_market_profile(market_id: str) -> MarketProfile:
    key = market_id.strip().upper()
    try:
        return _MARKETS[key]
    except KeyError as exc:
        supported = ", ".join(sorted(_MARKETS))
        raise ValueError(
            f"Unsupported market {market_id!r}; registered markets: {supported}"
        ) from exc


def registered_markets() -> tuple[str, ...]:
    return tuple(sorted(_MARKETS))


register_market(
    MarketProfile(
        market_id="CN",
        timezone="Asia/Shanghai",
        currency="CNY",
        mic_to_suffix={"XSHG": "SH", "XSHE": "SZ", "XBSE": "BJ"},
        local_code_pattern=re.compile(r"\d{6}"),
        default_board_lot=100,
    )
)
register_market(
    MarketProfile(
        market_id="HK",
        timezone="Asia/Hong_Kong",
        currency="HKD",
        mic_to_suffix={"XHKG": "HK"},
        local_code_pattern=re.compile(r"\d{5}"),
        # HK board lots vary by instrument and must be loaded from master data.
        default_board_lot=None,
    )
)
