"""Host-owned runtime paths, independent of the Python installation directory."""

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path


def _setting(config: Path, settings: dict[str, object], key: str) -> Path | None:
    value = settings.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{config}: {key} must be a non-empty path string")
    # Relative entries belong to the config file, never to the caller's cwd.
    return config.parent / Path(value).expanduser()


@dataclass(frozen=True)
class RuntimePaths:
    state: Path
    credentials: Path

    @classmethod
    def load(cls) -> "RuntimePaths":
        config = Path.home() / ".config" / "lite-pi" / "runtime.toml"
        settings: dict[str, object] = {}
        if config.exists():
            try:
                settings = tomllib.loads(config.read_text())
            except tomllib.TOMLDecodeError as error:
                raise ValueError(f"{config}: invalid TOML: {error}") from error
        home = os.environ.get("LITE_PI_HOME")
        state = Path(home) if home else _setting(config, settings, "state_dir") or Path("~/.lite-pi")
        env_file = os.environ.get("LITE_PI_ENV_FILE")
        credentials = Path(env_file) if env_file else _setting(config, settings, "env_file") or state / ".env"
        return cls(state.expanduser().resolve(), credentials.expanduser().resolve())


PATHS = RuntimePaths.load()
