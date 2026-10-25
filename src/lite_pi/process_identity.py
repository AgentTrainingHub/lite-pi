"""Identify a local process across PID reuse; unverifiable state never proves exit."""

import os
import subprocess
import sys
from pathlib import Path


def process_identity(pid: int) -> str | None:
    if sys.platform == "linux":
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
        except FileNotFoundError:
            return None
        # comm is parenthesized and can itself contain spaces and parentheses.
        fields = stat.rsplit(")", 1)[-1].split()
        if len(fields) < 20 or not fields[19].isdigit():
            raise OSError("cannot verify process identity: invalid proc stat")
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        if not boot:
            raise OSError("cannot verify process identity: empty boot id")
        return f"linux:{boot}:{fields[19]}"
    try:
        result = subprocess.run(
            ["/bin/ps", "-p", str(pid), "-o", "lstart="],
            capture_output=True,
            text=True,
            env={"PATH": "/usr/bin:/bin"},
            check=False,
            timeout=5,
        )
    except subprocess.TimeoutExpired as error:
        raise OSError("cannot verify process identity: ps timed out") from error
    if result.returncode == 1:
        return None
    if result.returncode != 0:
        raise OSError("cannot verify process identity")
    return result.stdout.strip() or None


def process_alive(pid: int, identity: str | None = None) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    if sys.platform == "linux" and identity is not None and not identity.startswith("linux:"):
        # Legacy ps timestamps cannot be compared with proc start ticks. Keep a
        # live PID's slot until it exits; a format change is not proof of death.
        return True
    if identity is None:
        return True
    try:
        return process_identity(pid) == identity
    except OSError:
        # Releasing a slot needs proof of exit; an unreadable identity keeps it held.
        return True
