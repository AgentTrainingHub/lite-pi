"""Download the fixed, inert ONNX/tokenizer files; this script makes no model API call."""

import argparse
import json
import os
import urllib.request
from pathlib import Path

from lite_pi.retrieval.embedding import DEFAULT_MODEL_DIR, MODEL, REVISION, Representation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--proxy", help="Explicit HTTPS proxy for public model downloads")
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    handlers = [urllib.request.ProxyHandler({"https": args.proxy})] if args.proxy else []
    opener = urllib.request.build_opener(*handlers)
    files = {
        "model.onnx": "onnx/model.onnx",
        "tokenizer.json": "onnx/tokenizer.json",
        "config.json": "config.json",
    }
    for name, remote in files.items():
        target = args.directory / name
        if target.exists():
            print(f"present: {name}")
            continue
        partial = target.with_suffix(target.suffix + ".part")
        try:
            with (
                opener.open(
                    f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{remote}", timeout=60
                ) as source,
                partial.open("xb") as output,
            ):
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)
            os.replace(partial, target)
        finally:
            partial.unlink(missing_ok=True)
        print(f"downloaded: {name}", flush=True)
    metadata = {**Representation().payload(), "license": "MIT"}
    (args.directory / "representation.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(args.directory)


if __name__ == "__main__":
    main()
