"""Build the isolated HNDL variant from the same native product source tree."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="moss-hndl-build-") as directory:
        stage = Path(directory)
        for name in ("src", "LICENSES"):
            shutil.copytree(ROOT / name, stage / name,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for name in ("README.md", "LICENSE", "NOTICE"):
            shutil.copyfile(ROOT / name, stage / name)
        shutil.copyfile(ROOT / "profiles/hypergan-hndl/pyproject.toml", stage / "pyproject.toml")
        subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps",
                        "--no-build-isolation", "--wheel-dir", str(output), str(stage)], check=True)


if __name__ == "__main__":
    main()
