import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
os.chdir(root)
sys.path.append(str(root))

from src.utils.io import read_jsonl


ANNOTATION_PATH = Path("data/outputs/annotation/hard_subset_300_annotation_assist.jsonl")


def main() -> None:
    if not ANNOTATION_PATH.exists():
        raise FileNotFoundError(
            f"Annotation assist file not found: {ANNOTATION_PATH}. "
            "Run scripts/05_annotation_assist.py first."
        )

    records = read_jsonl(ANNOTATION_PATH)
    annotated_count = sum(1 for record in records if record.get("annotated") is True)
    print(f"Annotation file: {ANNOTATION_PATH}")
    print(f"Annotated records: {annotated_count}/{len(records)}")
    print("No automatic labels were changed. Manually finalize labels in the annotation file before rewrite analysis.")


if __name__ == "__main__":
    main()
