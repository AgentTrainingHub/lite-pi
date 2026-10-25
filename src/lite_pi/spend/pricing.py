"""Official DeepSeek peak prices used to reserve spend.

Fetched 2026-10-14 from https://api-docs.deepseek.com/zh-cn/quick_start/pricing/.
Reservations always use the higher weekday peak rates and cache-miss input price.
Idle-period and cache-hit prices are used only when a response reports them.
"""

from decimal import Decimal

MODEL = "deepseek-flash"
ENDPOINT = "https://api.deepseek.com"
ALLOWANCE_CAP_CNY = Decimal("15")
RESERVE_FLOOR_CNY = Decimal("10")
# Peak, per token.
INPUT_MISS_CNY = Decimal("2") / Decimal("1000000")
INPUT_HIT_CNY = Decimal("0.04") / Decimal("1000000")
OUTPUT_CNY = Decimal("8") / Decimal("1000000")
INPUT_ESTIMATE_MARGIN = Decimal("1.25")
CHARS_PER_TOKEN_ESTIMATE = 3
MAX_OUTPUT_TOKENS = 2048
MAX_IN_FLIGHT = 1
TEAM_MAX_IN_FLIGHT = 2
QUANT = Decimal("0.00000001")


def allowance_cny(balance_cny: Decimal) -> Decimal:
    raw = balance_cny - RESERVE_FLOOR_CNY
    if raw <= 0:
        return Decimal("0")
    if raw >= ALLOWANCE_CAP_CNY:
        return ALLOWANCE_CAP_CNY
    return raw


def q(value: Decimal) -> str:
    return str(value.quantize(QUANT))
