"""Run the tool real-model acceptance group."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from acceptance import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main("tools"))
