"""Bootstrap and launch the constitutional-ai GUI with minimal user setup.

This helper is intended for non-technical users:
1) creates `.venv` if needed
2) installs the package in editable mode
3) starts the GUI server
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path


def _venv_bin(venv_dir: Path, name: str) -> Path:
    """Return an executable path inside the virtual environment."""
    if platform.system().lower().startswith("win"):
        return venv_dir / "Scripts" / f"{name}.exe"
    return venv_dir / "bin" / name


def _run(cmd: list[str], cwd: Path) -> None:
    """Run a subprocess command and raise on failure."""
    print(f"[launch_gui] $ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main(argv: list[str] | None = None) -> int:
    """Create venv, install package, and launch the GUI server."""
    args = argv or []
    repo_root = Path(__file__).resolve().parent
    venv_dir = repo_root / ".venv"

    if not venv_dir.exists():
        print("[launch_gui] Creating virtual environment...")
        _run([sys.executable, "-m", "venv", str(venv_dir)], cwd=repo_root)

    python_bin = _venv_bin(venv_dir, "python")
    pip_bin = _venv_bin(venv_dir, "pip")

    # Ensure pip is available and install the package + dependencies.
    _run([str(python_bin), "-m", "ensurepip", "--upgrade"], cwd=repo_root)
    _run([str(pip_bin), "install", "--disable-pip-version-check", "--upgrade", "pip"], cwd=repo_root)
    _run([str(pip_bin), "install", "--disable-pip-version-check", "-e", "."], cwd=repo_root)

    print("[launch_gui] Starting GUI server...")
    env = dict(os.environ)
    subprocess.run([str(python_bin), "-m", "constitutional_ai.server", *args], cwd=str(repo_root), check=True, env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
