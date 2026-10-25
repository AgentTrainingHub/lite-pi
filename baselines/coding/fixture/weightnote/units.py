"""Convert a numeric amount in a named unit to whole grams."""


def to_grams(amount: float, unit: str) -> int:
    if unit == "g":
        return int(round(amount))
    if unit == "kg":
        return int(round(amount * 1000))
    raise ValueError(unit)
