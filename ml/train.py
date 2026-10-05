"""Training entry point placeholder.

Move/import the user's existing training code here; this starter intentionally
will not create a production model without the original data and pipeline.
"""
from pathlib import Path


def main() -> None:
    artifact_dir = Path(__file__).resolve().parents[1] / "backend" / "models"
    print(f"Place the user's trained pipeline artifact in: {artifact_dir}")
    print("No model was trained: original training data/code were not supplied.")


if __name__ == "__main__":
    main()
