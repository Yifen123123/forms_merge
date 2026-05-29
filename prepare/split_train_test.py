from pathlib import Path
import json
import random
from collections import defaultdict


INPUT_JSON_PATH = Path("processed/classifier_dataset.json")

OUTPUT_TRAIN_JSON_PATH = Path("processed/train_dataset.json")
OUTPUT_TEST_JSON_PATH = Path("processed/test_dataset.json")
OUTPUT_ALL_WITH_SPLIT_PATH = Path("processed/classifier_dataset_with_split.json")

TEST_RATIO = 0.2
RANDOM_SEED = 42


def load_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=4
        )


def group_by_label(records: list[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)

    for record in records:
        label_name = record["label_name"]
        grouped[label_name].append(record)

    return grouped


def get_test_count(n: int, test_ratio: float) -> int:
    """
    根據每個類別的資料量決定 test 筆數。
    """
    if n <= 1:
        return 0

    if 2 <= n <= 4:
        return 1

    return max(1, round(n * test_ratio))


def split_dataset(
    records: list[dict],
    test_ratio: float = 0.2,
    random_seed: int = 42
) -> tuple[list[dict], list[dict], list[dict]]:
    random.seed(random_seed)

    grouped = group_by_label(records)

    train_records = []
    test_records = []
    all_with_split = []

    for label_name, items in grouped.items():
        items = items.copy()
        random.shuffle(items)

        n = len(items)
        test_count = get_test_count(n, test_ratio)

        test_items = items[:test_count]
        train_items = items[test_count:]

        for item in train_items:
            new_item = item.copy()
            new_item["split"] = "train"
            train_records.append(new_item)
            all_with_split.append(new_item)

        for item in test_items:
            new_item = item.copy()
            new_item["split"] = "test"
            test_records.append(new_item)
            all_with_split.append(new_item)

    train_records.sort(
        key=lambda x: (
            x["doc_type"],
            x["unit"],
            x["category"],
            x["call_id"]
        )
    )

    test_records.sort(
        key=lambda x: (
            x["doc_type"],
            x["unit"],
            x["category"],
            x["call_id"]
        )
    )

    all_with_split.sort(
        key=lambda x: (
            x["doc_type"],
            x["unit"],
            x["category"],
            x["call_id"]
        )
    )

    return train_records, test_records, all_with_split


def print_split_summary(
