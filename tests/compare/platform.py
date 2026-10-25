"""Record the current machine. Docker output is not a Linux host or WSL2 result."""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import subprocess
from pathlib import Path


def host_facts() -> dict[str, str]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }


def docker_server() -> dict[str, str]:
    if shutil.which("docker") is None:
        return {"status": "absent"}
    completed = subprocess.run(
        ["docker", "version", "--format", "{{.Server.Version}}"],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if completed.returncode != 0:
        return {"status": "unavailable"}
    return {"status": "present", "server": completed.stdout.strip()}


def wsl_available() -> bool:
    return platform.system() == "Windows" or shutil.which("wsl.exe") is not None


def fresh_import(venv: Path, repo: Path) -> dict[str, str]:
    """Install the checkout into a new virtualenv and import it from a temp home."""
    python = venv / "bin" / "python"
    env = {key: os.environ[key] for key in ("PATH", "HOME", "TMPDIR") if key in os.environ}
    env["UV_PROJECT_ENVIRONMENT"] = str(venv)
    subprocess.run(
        ["uv", "sync", "--project", str(repo), "--locked", "--no-dev", "--no-default-groups"],
        check=True,
        capture_output=True,
        env=env,
        timeout=180,
    )
    home = venv.parent / "home"
    home.mkdir(exist_ok=True)
    completed = subprocess.run(
        [str(python), "-c", "import lite_pi,platform; print(platform.python_version())"],
        check=False,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "HOME": str(home), "PYTHONDONTWRITEBYTECODE": "1"},
    )
    return {
        "returncode": str(completed.returncode),
        "python": completed.stdout.strip(),
        "home": str(home),
        "lock_sha256": hashlib.sha256((repo / "uv.lock").read_bytes()).hexdigest(),
        "install_command": "uv sync --locked --no-dev --no-default-groups",
    }
