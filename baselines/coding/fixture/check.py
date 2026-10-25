from weightnote.invoice import total_grams


CASES = [
    (["200 g", "1 kg"], 1200),
    (["2 kg"], 2000),
    (["10 g"], 10),
]


def main() -> int:
    failed = 0
    for lines, expected in CASES:
        got = total_grams(lines)
        if got != expected:
            print(f"FAIL {lines!r}: got {got} expected {expected}")
            failed += 1
        else:
            print(f"OK {lines!r}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
