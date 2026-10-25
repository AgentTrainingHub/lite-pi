"""Descriptor-relative access keeps symlinks out of file-tool authority."""

import hashlib
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from uuid import uuid4

from lite_pi.runtime import PATHS
from lite_pi.storage import locked
from lite_pi.tools.contracts import FileVersion

LOCKS = PATHS.state / "file-locks"
PROTECTED = {".git", ".artifacts", ".ssh", ".aws", ".env"}


def protected(name: str) -> bool:
    return name.casefold() in PROTECTED or name.casefold().startswith(".env.")


_HOST_FILES = {
    "extensions",
    "mcp.json",
    "corpus.json",
    "knowledge.sqlite",
    "knowledge.sqlite-wal",
    "knowledge.sqlite-shm",
    "knowledge.sqlite-journal",
}


def executable_config(path: str) -> bool:
    parts = tuple(part.casefold() for part in Path(path).parts)
    return bool(parts and parts[0] == ".lite-pi" and (len(parts) == 1 or parts[1] in _HOST_FILES))


class PublishedFileError(RuntimeError):
    """The destination was published, but durability or cleanup failed."""


class Workspace:
    def __init__(self, root: Path):
        self.root = root.resolve(strict=True)
        self.observed: dict[str, str] = {}

    def parts(self, path: str) -> tuple[str, ...]:
        raw = Path(path)
        if raw.is_absolute():
            try:
                raw = raw.relative_to(self.root)
            except ValueError as error:
                raise ValueError("path escapes the project") from error
        parts = raw.parts
        if not parts or ".." in parts or path.startswith("~"):
            raise ValueError("path escapes the project")
        if any(protected(part) for part in parts):
            raise ValueError("protected project path")
        return parts

    @contextmanager
    def parent(self, path: str, *, create: bool = False) -> Iterator[tuple[int, str]]:
        parts = self.parts(path)
        descriptor = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for part in parts[:-1]:
                if create:
                    with suppress(FileExistsError):
                        os.mkdir(part, dir_fd=descriptor)
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
            yield descriptor, parts[-1]
        finally:
            os.close(descriptor)

    def read(self, path: str, *, remember: bool = True, max_bytes: int | None = None) -> bytes:
        with self.parent(path) as (directory, name):
            data, _ = self._read(directory, name, max_bytes=max_bytes)
        if remember:
            self.observed["/".join(self.parts(path)).casefold()] = hashlib.sha256(data).hexdigest()
        return data

    def directories(self, path: str) -> list[str]:
        """List child directories without following links in any path component."""
        with self.parent(path) as (parent, name):
            descriptor = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            try:
                found = []
                for child in sorted(os.listdir(descriptor)):
                    info = os.stat(child, dir_fd=descriptor, follow_symlinks=False)
                    if stat.S_ISLNK(info.st_mode):
                        raise ValueError(f"symlink directory entry: {path}/{child}")
                    if stat.S_ISDIR(info.st_mode) and not child.startswith("."):
                        found.append(child)
                return found
            finally:
                os.close(descriptor)

    @staticmethod
    def _read(directory: int, name: str, *, max_bytes: int | None = None) -> tuple[bytes, os.stat_result]:
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        with os.fdopen(descriptor, "rb") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError("only regular files with one link are supported")
            data = handle.read() if max_bytes is None else handle.read(max_bytes + 1)
            if max_bytes is not None and len(data) > max_bytes:
                raise ValueError(f"resource exceeds {max_bytes} bytes")
            return data, info

    @contextmanager
    def mutation(self, path: str) -> Iterator[None]:
        if executable_config("/".join(self.parts(path))):
            raise ValueError("host executable configuration is read-only to tools")
        normalized = str(self.root.joinpath(*self.parts(path)))
        key = hashlib.sha256(normalized.casefold().encode()).hexdigest()
        with locked(LOCKS / (key + ".lock")):
            yield

    def remove(self, path: str, *, expected: bytes | None) -> None:
        with self.mutation(path), self.parent(path) as (directory, name):
            if expected is None:
                os.rmdir(name, dir_fd=directory)
            else:
                current, _ = self._read(directory, name)
                if current != expected:
                    raise ValueError("host file changed before removal")
                os.unlink(name, dir_fd=directory)
            os.fsync(directory)

    def replace(
        self,
        path: str,
        data: bytes,
        *,
        expected: bytes | None,
        overwrite: bool,
        mode: int | None = None,
        expected_file: FileVersion | None = None,
    ) -> None:
        if executable_config("/".join(self.parts(path))):
            raise ValueError("host executable configuration is read-only to tools")
        published = False
        try:
            with self.parent(path, create=expected is None) as (directory, name):
                current = None
                target_mode = 0o644 if mode is None else mode
                try:
                    current, info = self._read(directory, name)
                    target_mode = stat.S_IMODE(info.st_mode) if mode is None else mode
                except FileNotFoundError:
                    pass
                if expected_file is not None:
                    digest = None if current is None else hashlib.sha256(current).hexdigest()
                    if digest != expected_file.sha256:
                        raise ValueError("approved file version changed; file unchanged")
                if current is not None and not overwrite:
                    raise ValueError(f"write refused: {path} exists; pass overwrite to replace it")
                if expected is not None and current != expected:
                    raise ValueError("file changed before write; file unchanged")
                observed = self.observed.get("/".join(self.parts(path)).casefold())
                if observed is not None and (
                    current is None or hashlib.sha256(current).hexdigest() != observed
                ):
                    raise ValueError("file changed since read; read it again before editing")
                temporary = ".lite-pi-" + uuid4().hex
                descriptor = os.open(
                    temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, target_mode, dir_fd=directory
                )
                try:
                    with os.fdopen(descriptor, "wb") as handle:
                        os.fchmod(handle.fileno(), target_mode)
                        handle.write(data)
                        handle.flush()
                        os.fsync(handle.fileno())
                    if current is None:
                        os.link(
                            temporary, name, src_dir_fd=directory, dst_dir_fd=directory, follow_symlinks=False
                        )
                        published = True
                    else:
                        latest, latest_info = self._read(directory, name)
                        if latest != current or latest_info.st_ino != info.st_ino:
                            raise ValueError("file changed before write; file unchanged")
                        os.replace(temporary, name, src_dir_fd=directory, dst_dir_fd=directory)
                        published = True
                    os.fsync(directory)
                finally:
                    with suppress(FileNotFoundError):
                        os.unlink(temporary, dir_fd=directory)
        except OSError as error:
            if published:
                raise PublishedFileError(f"{path}: {type(error).__name__}: {error}") from error
            raise
        self.observed["/".join(self.parts(path)).casefold()] = hashlib.sha256(data).hexdigest()
