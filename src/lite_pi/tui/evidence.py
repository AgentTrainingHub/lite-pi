"""Explicit detail expansion reads only host-owned evidence below the runs directory."""

import json
import re
from pathlib import Path

from lite_pi.session import RUNS
from lite_pi.tools.workspace import Workspace


def original(path: str) -> str:
    text = ""
    try:
        relative = Path(path).relative_to(RUNS)
        if len(relative.parts) != 3 or not re.fullmatch(r"run-[0-9a-f]{32}", relative.parts[0]):
            raise ValueError("invalid run evidence identity")
        if relative.parts[1] != "tool-raw" or relative.suffix != ".json":
            raise ValueError("not a raw tool receipt")
        reader = Workspace(RUNS)
        receipt = json.loads(reader.read(str(relative), remember=False, max_bytes=64 * 1024).decode())
        if not isinstance(receipt, dict) or not isinstance(receipt.get("result"), dict):
            raise ValueError("invalid original tool receipt")
        text = "原始工具输出\n" + json.dumps(receipt, ensure_ascii=False, indent=2)
        output = receipt.get("result", {}).get("output_path")
        if output is not None and not isinstance(output, str):
            raise ValueError("invalid process evidence path")
        if output:
            result_path = Path(output).relative_to(RUNS)
            if len(result_path.parts) < 2 or result_path.parts[0] != relative.parts[0]:
                raise ValueError("output belongs to another run")
            text += "\n原始进程输出\n" + reader.read(
                str(result_path), remember=False, max_bytes=64 * 1024
            ).decode("utf-8", "replace")
        return text
    except (OSError, ValueError, UnicodeError) as error:
        return f"原始证据不可读 · {type(error).__name__}: {error}\n完整原始证据：{path}\n" + text
