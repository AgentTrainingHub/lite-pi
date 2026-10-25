RATES = {"standard": 10}


def percent_off(name: str) -> int:
    if name not in RATES:
        raise KeyError(name)
    return RATES[name]
