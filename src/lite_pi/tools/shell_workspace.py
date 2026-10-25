"""A filtered command workspace prevents host path aliases from exposing credentials."""

import os
import stat
from dataclasses import dataclass, replace
from pathlib import Path

from lite_pi.tools.workspace import Workspace, protected

MAX_WORKSPACE_BYTES = 64 * 1024 * 1024
MAX_WORKSPACE_ENTRIES = 10000


@dataclass(frozen=True)
class Entry:
    kind: str
    mode: int
    content: bytes = b""
    target: str = ""


def snapshot(root: Path, *, reject_private: bool = False) -> dict[str, Entry]:
    workspace = Workspace(root)
    entries = {}
    total = 0

    def visit(directory: Path) -> None:
        nonlocal total
        for path in sorted(directory.iterdir()):
            if protected(path.name) or (directory == root and path.name.casefold() == ".lite-pi"):
                if reject_private:
                    raise ValueError("command created a protected project path")
                continue
            name = str(path.relative_to(root))
            info = path.lstat()
            mode = stat.S_IMODE(info.st_mode) & 0o777
            if stat.S_ISLNK(info.st_mode):
                entry = Entry("symlink", mode, target=os.readlink(path))
            elif stat.S_ISDIR(info.st_mode):
                entry = Entry("directory", mode)
            elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                if total + info.st_size > MAX_WORKSPACE_BYTES:
                    raise ValueError("command workspace exceeds 64 MiB")
                content = workspace.read(name, remember=False)
                total += len(content)
                if total > MAX_WORKSPACE_BYTES:
                    raise ValueError("command workspace exceeds 64 MiB")
                entry = Entry("file", mode, content)
            else:
                raise ValueError("command workspace requires single-link regular files and directories")
            entries[name] = entry
            if len(entries) > MAX_WORKSPACE_ENTRIES:
                raise ValueError("command workspace exceeds 10000 entries")
            if entry.kind == "directory":
                visit(path)

    visit(root)
    return entries


class ScopeRejected(ValueError):
    def __init__(self, paths: list[str]):
        self.paths = paths
        super().__init__("path is outside the current file scope")


def allowed_path(name: str, scope: tuple[str, ...]) -> bool:
    return any(name == prefix or name.startswith(prefix.rstrip("/") + "/") for prefix in scope)


class ShellWorkspace:
    def __init__(self, root: Path, path: Path):
        self.root, self.path = root, path
        self.before = snapshot(root)
        self.published: list[str] = []
        path.mkdir()
        for name, entry in self.before.items():
            destination = path / name
            if entry.kind == "directory":
                destination.mkdir()
            elif entry.kind == "symlink":
                destination.symlink_to(entry.target)
            else:
                destination.write_bytes(entry.content)
                destination.chmod(entry.mode)
        for name, entry in reversed(list(self.before.items())):
            if entry.kind == "directory":
                (path / name).chmod(entry.mode)

    def publish(self, guest_modes: dict[str, int], *, allowed_paths: tuple[str, ...] | None = None) -> None:
        after = snapshot(self.path, reject_private=True)
        if after.keys() != guest_modes.keys():
            raise ValueError("guest workspace inventory differs from host observation")
        after = {
            name: replace(entry, mode=guest_modes[name]) if entry.kind != "symlink" else entry
            for name, entry in after.items()
        }
        changed = [
            name for name in self.before.keys() | after.keys() if self.before.get(name) != after.get(name)
        ]
        # Validate the complete change before touching the host. Links are preserved as inputs only.
        for name in changed:
            before, updated = self.before.get(name), after.get(name)
            if (before and before.kind == "symlink") or (updated and updated.kind == "symlink"):
                raise ValueError("command changed a symlink; host project unchanged")
            if before and updated and before.kind != updated.kind:
                raise ValueError("command changed a path's file type; host project unchanged")
        if allowed_paths is not None:
            outside = [
                name
                for name in changed
                if (after.get(name) is None or after[name].kind != "directory")
                and not allowed_path(name, allowed_paths)
            ]
            if outside:
                raise ScopeRejected(outside)
        if snapshot(self.root) != self.before:
            raise ValueError("host project changed while command ran; command changes were not published")
        workspace = Workspace(self.root)
        directories = set()
        for name in changed:
            entry = after.get(name) or self.before[name]
            if entry.kind == "directory":
                directories.add(name)
            directories.update(str(parent) for parent in Path(name).parents if str(parent) != ".")
        prepared = []
        complete = False
        try:
            for name in sorted(directories, key=lambda value: (len(Path(value).parts), value)):
                entry = self.before.get(name) or after[name]
                with workspace.parent(name) as (directory, leaf):
                    if name not in self.before:
                        os.mkdir(leaf, entry.mode | 0o700, dir_fd=directory)
                        self.published.append(name)
                    _directory_mode(directory, leaf, entry.mode | 0o700)
                prepared.append(name)
            for name in sorted(changed):
                updated = after.get(name)
                before = self.before.get(name)
                if updated is None or updated.kind == "directory":
                    continue
                with workspace.mutation(name):
                    workspace.replace(
                        name,
                        updated.content,
                        expected=before.content if before else None,
                        overwrite=before is not None,
                        mode=updated.mode,
                    )
                self.published.append(name)
            for name in sorted(changed, key=lambda value: (-len(Path(value).parts), value)):
                if name in after:
                    continue
                before = self.before[name]
                workspace.remove(name, expected=before.content if before.kind == "file" else None)
                self.published.append(name)
            complete = True
        finally:
            for name in reversed(prepared):
                if name not in after and name in self.published:
                    continue
                final_entry = after.get(name) if complete else self.before.get(name) or after.get(name)
                if final_entry is not None:
                    with workspace.parent(name) as (directory, leaf):
                        _directory_mode(directory, leaf, final_entry.mode)
                    if complete and name in changed and name not in self.published:
                        self.published.append(name)


def _directory_mode(directory: int, name: str, mode: int) -> None:
    descriptor = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
    try:
        os.fchmod(descriptor, mode)
    finally:
        os.close(descriptor)
