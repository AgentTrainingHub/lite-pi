"""Prepare the official Python runtime once; persist its immutable image digest."""

import json
import subprocess

from lite_pi.storage import atomic_write
from lite_pi.tools.shell import DOCKER_ENV, IMAGE_FILE


def main() -> int:
    ready = subprocess.run(
        ["docker", "info", "--format", "{{.ServerVersion}}"],
        capture_output=True,
        text=True,
        timeout=15,
        env=DOCKER_ENV,
    )
    if ready.returncode:
        print(
            json.dumps(
                {"status": "blocked", "reason": "Docker engine unavailable", "detail": ready.stderr.strip()}
            )
        )
        return 2
    image = "python:3.12-slim"
    pulled = subprocess.run(["docker", "pull", image], timeout=300, env=DOCKER_ENV)
    if pulled.returncode:
        return pulled.returncode
    inspected = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{json .RepoDigests}}", image],
        capture_output=True,
        text=True,
        check=True,
        env=DOCKER_ENV,
    )
    digest = next(item for item in json.loads(inspected.stdout) if item.startswith("python@sha256:"))
    atomic_write(IMAGE_FILE, (json.dumps({"image": digest}) + "\n").encode())
    print(json.dumps({"status": "ready", "image": digest, "evidence": str(IMAGE_FILE)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
