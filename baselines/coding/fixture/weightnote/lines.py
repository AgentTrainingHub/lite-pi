"""Split a shipment line into a number and a unit label."""


def split_line(text: str) -> tuple[float, str]:
    parts = text.strip().split()
    if len(parts) != 2:
        raise ValueError(text)
    return float(parts[0]), parts[1].lower()
