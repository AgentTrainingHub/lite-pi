from rateline.quote import quote_total


def main() -> int:
    got = quote_total(cents=10_000, rate_name="standard")
    expected = 9_000
    if got != expected:
        print(f"FAIL quote_total: got {got} expected {expected}")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
