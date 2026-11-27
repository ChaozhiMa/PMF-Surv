"""Convenience entry point delegating to the canonical training script."""

from pathlib import Path
import runpy


def main() -> None:
    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts" / "train.py"),
        run_name="__main__",
    )


if __name__ == "__main__":
    main()
