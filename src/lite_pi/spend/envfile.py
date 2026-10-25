"""Read the local .env without exporting credentials into the process environment."""

from lite_pi.runtime import PATHS

CONFIG_NAMES = ("DEEPSEEK_API_KEY", "DEEPSEEK_API_BASE", "DEEPSEEK_MODEL")


def read_local_config() -> dict[str, str]:
    path = PATHS.credentials
    local: dict[str, str] = {}
    if path.exists():
        if path.stat().st_mode & 0o077:
            raise ValueError("credential file must have mode 0600")
        for line in path.read_text().splitlines():
            if not line.strip() or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            local[key] = value
    return local


def resolve_config() -> dict[str, str]:
    import os

    local = read_local_config()
    values = {name: os.environ.get(name, local.get(name, "")) for name in CONFIG_NAMES}
    return values
