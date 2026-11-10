"""
backend/services/kit_service.py

Turns a treatment step into a priced treatment kit: how much of the
TNAU-listed product the grower needs for their field, and which packs
to buy.

    dose (TNAU text)  "0.25% (2.5 g per litre of water)", "200 ml per acre"
      -> rate         2.5 g per litre of spray / 494 ml per hectare
      -> quantity     rate x spray volume (or area) x number of sprays
      -> packs        cheapest combination of catalog packs that covers it

Everything here is deterministic: the AI assistant may suggest the field
size and which listed option to use, but quantities and prices are always
computed by this module from the curated plan and the catalog, and the
server recomputes them when the order is created.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from functools import lru_cache, reduce
from pathlib import Path
from typing import Any, Dict, List, Sequence

KNOWLEDGE_DIR = Path(__file__).resolve().parents[1] / "knowledge"
CATALOG_FILE = KNOWLEDGE_DIR / "shop_catalog.json"

# Hectares per unit. A cent is 1/100 acre (common in South India).
AREA_UNITS: Dict[str, float] = {
    "acre": 0.404686,
    "hectare": 1.0,
    "cent": 0.00404686,
    "sqm": 0.0001,
}
AREA_UNIT_LABELS = {"acre": "acre", "hectare": "hectare", "cent": "cent", "sqm": "square metre"}

MAX_HECTARES = 50.0  # a smallholder shop, not a bulk distributor
MAX_SPRAYS = 4
DEFAULT_SPRAYS = 2

ACRE_HECTARES = AREA_UNITS["acre"]

_RATE = re.compile(
    r"(\d+(?:\.\d+)?)(?:\s*-\s*(\d+(?:\.\d+)?))?\s*(kg|g|ml)\s+per\s+(?:(\d+)\s+)?"
    r"(litres?|liters?|acre|hectare)",
    re.IGNORECASE,
)
_PERCENT = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*%")

CENT = Decimal("0.01")


class KitError(Exception):
    """Invalid kit request (bad field size, unknown product...)."""

    def __init__(self, message: str, code: str, http_status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.http_status = http_status


@dataclass(frozen=True)
class DoseRate:
    """Product needed per litre of spray, or per hectare of field."""

    amount: float  # in the catalog unit (g or ml)
    basis: str  # "litre" | "hectare"


@dataclass(frozen=True)
class Field:
    area: float
    area_unit: str
    sprays: int

    @property
    def hectares(self) -> float:
        return self.area * AREA_UNITS[self.area_unit]

    def to_dict(self) -> Dict[str, Any]:
        return {"area": self.area, "area_unit": self.area_unit, "sprays": self.sprays}


@lru_cache(maxsize=1)
def load_catalog() -> Dict[str, Any]:
    return json.loads(CATALOG_FILE.read_text(encoding="utf-8"))


def catalog_meta() -> Dict[str, Any]:
    meta = load_catalog()["_meta"]
    return {k: meta[k] for k in ("seller", "currency", "prices_note", "spray_litres_note")}


def currency() -> str:
    return load_catalog()["_meta"]["currency"]


def spray_litres_per_hectare() -> float:
    return float(load_catalog()["_meta"]["spray_litres_per_hectare"])


def catalog_entry(product_name: str) -> Dict[str, Any] | None:
    return load_catalog()["products"].get(product_name)


def parse_dose(text: str) -> DoseRate | None:
    """
    Rate from a TNAU dose text, or None if it cannot be read.

    Ranges use the upper value so the kit is never too small. A bare
    percentage means weight per volume (1% = 10 g per litre).
    """

    match = _RATE.search(text or "")

    if match:
        amount = float(match.group(2) or match.group(1))
        if match.group(3).lower() == "kg":
            amount *= 1000
        amount /= float(match.group(4) or 1)
        per = match.group(5).lower()

        if per.startswith(("litre", "liter")):
            return DoseRate(amount, "litre")
        if per == "acre":
            return DoseRate(amount / ACRE_HECTARES, "hectare")
        return DoseRate(amount, "hectare")

    match = _PERCENT.match(text or "")

    if match:
        return DoseRate(float(match.group(1)) * 10, "litre")

    return None


def validate_field(area: Any, area_unit: Any, sprays: Any) -> Field:
    """Field size and number of sprays from a request."""

    try:
        area_value = float(area)
    except (TypeError, ValueError):
        raise KitError("Enter the size of the area to treat.", "INVALID_AREA") from None

    if not math.isfinite(area_value) or area_value <= 0:
        raise KitError("Enter the size of the area to treat.", "INVALID_AREA")

    if area_unit not in AREA_UNITS:
        raise KitError("Choose acre, hectare, cent or square metre.", "INVALID_AREA_UNIT")

    if area_value * AREA_UNITS[area_unit] > MAX_HECTARES:
        raise KitError(
            f"The shop handles up to {MAX_HECTARES:g} hectares per order. For larger "
            "farms, contact your agro-input dealer.",
            "AREA_TOO_LARGE",
        )

    if sprays in (None, ""):
        sprays = DEFAULT_SPRAYS

    try:
        spray_count = int(sprays)
    except (TypeError, ValueError):
        raise KitError("Number of sprays must be a whole number.", "INVALID_SPRAYS") from None

    if isinstance(sprays, bool) or not 1 <= spray_count <= MAX_SPRAYS:
        raise KitError(f"Number of sprays must be 1 to {MAX_SPRAYS}.", "INVALID_SPRAYS")

    return Field(round(area_value, 4), area_unit, spray_count)


def money(value: float | Decimal) -> Decimal:
    return Decimal(str(value)).quantize(CENT, rounding=ROUND_HALF_UP)


def pack_price(entry: Dict[str, Any], size: int) -> Decimal:
    premium = load_catalog()["_meta"]["pack_premium"].get(str(size), 1.0)
    return max(money(entry["price_per_unit"] * size / 1000 * premium), Decimal("0.50"))


def pack_label(entry: Dict[str, Any], size: int) -> str:
    unit = entry["unit"]

    if size >= 1000:
        text = f"{size / 1000:g} {'kg' if unit == 'g' else 'litre'}"
    else:
        text = f"{size} {unit}"

    return f"{text} {entry['pack_label']}" if entry.get("pack_label") else f"{text} pack"


def choose_packs(entry: Dict[str, Any], needed: float) -> Dict[int, int]:
    """Cheapest pack combination holding at least `needed` (pack size -> count)."""

    sizes = sorted(int(s) for s in entry["packs"])
    step = reduce(math.gcd, sizes)
    target = max(1, math.ceil(round(needed, 6) / step))
    limit = target + sizes[-1] // step
    prices = {s: pack_price(entry, s) for s in sizes}

    # cost[u] = cheapest way to hold exactly u * step (None = impossible).
    cost: List[Decimal | None] = [None] * (limit + 1)
    last: List[int] = [0] * (limit + 1)
    cost[0] = Decimal(0)

    for units in range(1, limit + 1):
        for size in sizes:
            prev = units - size // step
            if prev >= 0 and cost[prev] is not None:
                candidate = cost[prev] + prices[size]
                if cost[units] is None or candidate < cost[units]:
                    cost[units] = candidate
                    last[units] = size

    best = min(
        (u for u in range(target, limit + 1) if cost[u] is not None),
        key=lambda u: (cost[u], u),
    )

    counts: Dict[int, int] = {}
    while best > 0:
        size = last[best]
        counts[size] = counts.get(size, 0) + 1
        best -= size // step

    return dict(sorted(counts.items(), reverse=True))


def amount_text(amount: float, unit: str) -> str:
    if amount >= 1000:
        return f"{amount / 1000:.2f}".rstrip("0").rstrip(".") + (" kg" if unit == "g" else " litres")
    return f"{math.ceil(amount)} {unit}"


def product_quote(product: Dict[str, str], field: Field) -> Dict[str, Any] | None:
    """Kit for one listed product, or None if it is not sold / dose unreadable."""

    entry = catalog_entry(product["name"])
    rate = parse_dose(product["dose"])

    if entry is None or rate is None:
        return None

    spray_litres = field.hectares * spray_litres_per_hectare()
    per_spray = rate.amount * (spray_litres if rate.basis == "litre" else field.hectares)
    needed = per_spray * field.sprays
    counts = choose_packs(entry, needed)

    items = []
    for size, count in counts.items():
        price = pack_price(entry, size)
        items.append({
            "sku": f"{entry['sku']}-{size}",
            "name": f"{product['name']} - {pack_label(entry, size)}",
            "product": product["name"],
            "pack_size": size,
            "unit": entry["unit"],
            "quantity": count,
            "unit_price": str(price),
            "line_total": str(price * count),
        })

    total = sum((Decimal(i["line_total"]) for i in items), Decimal(0))
    packed = sum(size * count for size, count in counts.items())

    return {
        "product": product["name"],
        "kind": product.get("kind"),
        "group": product.get("group"),
        "dose": product["dose"],
        "unit": entry["unit"],
        "rate_basis": rate.basis,
        "spray_litres": round(spray_litres, 1) if rate.basis == "litre" else None,
        "per_spray": round(per_spray, 1),
        "needed": round(needed, 1),
        "needed_text": amount_text(needed, entry["unit"]),
        "packed_text": amount_text(packed, entry["unit"]),
        "items": items,
        "total": str(money(total)),
        "currency": currency(),
        "note": entry.get("note"),
    }


def kit_options(step_products: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    """Listed products the shop can sell (in catalog, readable dose)."""

    return [
        p for p in step_products
        if catalog_entry(p["name"]) is not None and parse_dose(p["dose"]) is not None
    ]


def build_quote(
    step_products: Sequence[Dict[str, str]],
    field: Field,
    product_name: str | None = None,
) -> Dict[str, Any]:
    """
    Kit for a step: the chosen listed product (default: the cheapest),
    plus the price of each alternative the step lists.
    """

    quotes = [q for q in (product_quote(p, field) for p in kit_options(step_products)) if q]

    if not quotes:
        raise KitError(
            "None of this step's products can be ordered in the app.",
            "NOTHING_TO_ORDER",
        )

    if product_name:
        chosen = next((q for q in quotes if q["product"] == product_name), None)
        if chosen is None:
            raise KitError("Choose one of the products listed for this step.", "INVALID_PRODUCT")
    else:
        chosen = min(quotes, key=lambda q: Decimal(q["total"]))

    area_label = AREA_UNIT_LABELS[field.area_unit]

    return {
        **chosen,
        "field": field.to_dict(),
        "field_text": f"{field.area:g} {area_label}{'' if field.area == 1 else 's'}, "
        f"{field.sprays} spray{'' if field.sprays == 1 else 's'}",
        "alternatives": [
            {"product": q["product"], "total": q["total"], "group": q["group"]}
            for q in quotes
            if q["product"] != chosen["product"]
        ],
    }
