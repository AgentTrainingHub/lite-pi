"""Sum shipment lines into grams."""

from weightnote.lines import split_line
from weightnote.units import to_grams


def total_grams(lines: list[str]) -> int:
    total = 0
    for line in lines:
        amount, _unit = split_line(line)
        total += to_grams(amount, "g")
    return total
