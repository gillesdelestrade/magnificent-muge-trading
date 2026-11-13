"""Fee model: brackets per order amount + loading/saving of the broker profiles."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml

from ..paths import FEE_PROFILES_PATH

PEA_CAP_RATE = 0.005  # legal cap for online orders in a PEA
STALE_AFTER_DAYS = 183  # ~6 months


@dataclass(frozen=True)
class FeeBracket:
    up_to: float | None = None   # inclusive upper bound of the order amount; None = no limit
    fixed: float = 0.0           # euros
    pct: float = 0.0             # percent of the amount (0.09 means 0.09 %)
    minimum: float | None = None
    maximum: float | None = None

    def fee(self, amount: float) -> float:
        value = self.fixed + self.pct / 100.0 * amount
        if self.minimum is not None:
            value = max(self.minimum, value)
        if self.maximum is not None:
            value = min(self.maximum, value)
        return value

    def to_dict(self) -> dict:
        out: dict = {}
        if self.up_to is not None:
            out["jusqu_a"] = self.up_to
        out["fixe"] = self.fixed
        out["pourcentage"] = self.pct
        if self.minimum is not None:
            out["minimum"] = self.minimum
        if self.maximum is not None:
            out["maximum"] = self.maximum
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "FeeBracket":
        def num(key):
            v = d.get(key)
            return None if v is None else float(v)
        return cls(up_to=num("jusqu_a"), fixed=num("fixe") or 0.0, pct=num("pourcentage") or 0.0,
                   minimum=num("minimum"), maximum=num("maximum"))


@dataclass(frozen=True)
class FeeProfile:
    name: str
    brackets: tuple[FeeBracket, ...] = (FeeBracket(),)
    description: str = ""
    free_orders_per_month: int = 0
    free_orders_max_amount: float = 0.0
    account_type: str = "CTO"
    verified_on: date | None = None
    source_url: str = ""

    def __post_init__(self):
        ordered = sorted(self.brackets, key=lambda b: float("inf") if b.up_to is None else b.up_to)
        object.__setattr__(self, "brackets", tuple(ordered))

    def brokerage(self, amount: float) -> float:
        """Brokerage fee of one order, ignoring the monthly free orders."""
        if amount <= 0 or not self.brackets:
            return 0.0
        bracket = next((b for b in self.brackets if b.up_to is None or amount <= b.up_to),
                       self.brackets[-1])
        value = bracket.fee(amount)
        if self.account_type.upper() == "PEA":
            value = min(value, PEA_CAP_RATE * amount)
        return round(max(value, 0.0), 2)

    def is_free_eligible(self, amount: float) -> bool:
        return self.free_orders_per_month > 0 and amount <= self.free_orders_max_amount

    def is_stale(self, today: date | None = None) -> bool:
        if self.verified_on is None:
            return True
        return ((today or date.today()) - self.verified_on).days > STALE_AFTER_DAYS

    def to_dict(self) -> dict:
        out = {
            "name": self.name,
            "description": self.description,
            "account_type": self.account_type,
            "verified_on": self.verified_on.isoformat() if self.verified_on else None,
            "source_url": self.source_url,
        }
        if self.free_orders_per_month:
            out["free_orders_per_month"] = self.free_orders_per_month
            out["free_orders_max_amount"] = self.free_orders_max_amount
        out["brackets"] = [b.to_dict() for b in self.brackets]
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "FeeProfile":
        verified = d.get("verified_on")
        if isinstance(verified, str):
            verified = date.fromisoformat(verified)
        return cls(
            name=str(d["name"]),
            description=" ".join(str(d.get("description") or "").split()),
            brackets=tuple(FeeBracket.from_dict(b) for b in d.get("brackets") or [{}]),
            free_orders_per_month=int(d.get("free_orders_per_month") or 0),
            free_orders_max_amount=float(d.get("free_orders_max_amount") or 0.0),
            account_type=str(d.get("account_type") or "CTO").upper(),
            verified_on=verified,
            source_url=str(d.get("source_url") or ""),
        )


NO_FEES = FeeProfile(name="Sans frais", description="0 €")


@dataclass(frozen=True)
class Fees:
    brokerage: float = 0.0
    ttf: float = 0.0

    @property
    def total(self) -> float:
        return round(self.brokerage + self.ttf, 2)


def is_ttf_eligible(ticker: str) -> bool:
    """Simplification: the French financial transaction tax applies to Paris-listed shares.

    (In reality: French companies above 1 bn € of market capitalisation.)
    """
    return ticker.upper().endswith(".PA")


@dataclass
class FeeCalculator:
    """Fees of each order, with the monthly counter of free orders (reset every month)."""
    profile: FeeProfile = NO_FEES
    ttf_enabled: bool = False
    ttf_rate: float = 0.004
    _free_used: dict[tuple[int, int], int] = field(default_factory=dict)

    def compute(self, amount: float, day: date, side: str, ticker: str = "",
                commit: bool = False) -> Fees:
        """Fees of an order. ``commit=True`` records it (consumes a free order if any)."""
        brokerage = self.profile.brokerage(amount)
        if amount > 0 and self.profile.is_free_eligible(amount):
            key = (day.year, day.month)
            used = self._free_used.get(key, 0)
            if used < self.profile.free_orders_per_month:
                brokerage = 0.0
                if commit:
                    self._free_used[key] = used + 1
        ttf = 0.0
        if self.ttf_enabled and side == "buy" and is_ttf_eligible(ticker):
            ttf = round(self.ttf_rate * amount, 2)
        return Fees(brokerage=brokerage, ttf=ttf)


def load_profiles(path: Path | None = None) -> list[FeeProfile]:
    path = path or FEE_PROFILES_PATH
    if not path.exists():
        return [NO_FEES]
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    profiles = [FeeProfile.from_dict(p) for p in raw.get("profiles", [])]
    return profiles or [NO_FEES]


def save_profiles(profiles: list[FeeProfile], path: Path | None = None) -> None:
    path = path or FEE_PROFILES_PATH
    header = (
        "# Profils de frais des plateformes — modifiable à la main ou depuis l'onglet Paramétrage.\n"
        "# Par tranche : frais = min(maximum, max(minimum, fixe + pourcentage × montant))\n"
        "# pourcentage en % du montant (0.09 signifie 0,09 %) ; jusqu_a absent = sans plafond.\n"
        "# account_type : CTO ou PEA (en PEA, frais plafonnés à 0,5 % du montant de l'ordre).\n"
    )
    body = yaml.safe_dump({"profiles": [p.to_dict() for p in profiles]},
                          allow_unicode=True, sort_keys=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + body, encoding="utf-8")


def get_profile(name: str, profiles: list[FeeProfile] | None = None) -> FeeProfile:
    profiles = profiles if profiles is not None else load_profiles()
    return next((p for p in profiles if p.name == name), profiles[0])
