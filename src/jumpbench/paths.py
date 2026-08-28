from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Return the checkout root (parent of src/ or the cwd fallback)."""
    here = Path(__file__).resolve()
    for candidate in (here.parents[2], here.parents[1], Path.cwd()):
        if (candidate / "configs" / "models.yaml").exists():
            return candidate
    return Path.cwd()


def resolve(path: str | Path, root: Path | None = None) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (root or repo_root()) / path
