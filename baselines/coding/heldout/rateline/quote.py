from rateline.rates import percent_off


def quote_total(cents: int, rate_name: str) -> int:
    discount = percent_off(rate_name)
    once = cents - cents * discount // 100
    return once - once * discount // 100
