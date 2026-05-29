from pathlib import Path
import json
import random
import shutil
from collections import defaultdict


INPUT_DATA_DIR = Path("data_merged")
TRAIN_DATA_DIR = Path("train_data")
TEST_DATA_DIR = Path("test_data")

OUTPUT_ALL_JSON = Path("processed/classifier_dataset_split.json")
OUTPUT_TRAIN_JSON = Path("processed/train_dataset.json")
OUTPUT_TEST_JSON = Path("processed/test_dataset.json")

TEST_RATIO = 0.2
RANDOM_SEED = 42


def collect_files(data_dir: Path) -> dict:
    """
    依照 label_name 分組：
    {
        "會辦單位A_類別B": [
            {
                "path": Path(...),
                "call_id": "001",
                "doc_type": "行政會辦單",
                "unit": "會辦單位A",
                "category": "類別B",
                "label_name": "會辦單位A_類別B"
            }
        ]
    }
    """
    grouped = defaultdict(list)

    for txt_file in data_dir.rglob("*.txt"):
        relative_parts = txt_file.relative_to(data_dir).parts

        if len(relative_parts) != 4:
            print(f"略過格式不符檔案：{txt_file}")
            continue

        doc_type, unit, category, filename = relative_parts
        call_id = Path(filename).stem
        label_name = f"{unit}_{category}"

        item = {
            "path": txt_file,
            "call_id": call_id,
            "doc_type": doc_type,
            "unit": unit,
            "category": category,
            "label_name": label_name,
        }

        grouped[label_name].append(item)

    return grouped


def decide_split_count(n: int) -> int:
    """
    回傳 test 數量
    """
    if n <= 1:
        return 0

    if 2 <= n <= 4:
        return 1

    return max(1, round(n * TEST_RATIO))


def copy_file(item: dict, split: str) -> Path:
    source_path = item["path"]

    if split == "train":
        output_root = TRAIN_DATA_DIR
    elif split == "test":
        output_root = TEST_DATA_DIR
    else:
        raise ValueError(f"未知 split：{split}")

    target_path = (
        output_root
        / item["doc_type"]
        / item["unit"]
        / item["category"]
        / source_path.name
    )

    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists():
        raise FileExistsError(
            f"目標檔案已存在，為避免覆蓋已停止：\n"
            f"來源：{source_path}\n"
            f"目標：{target_path}"
        )

    shutil.copy2(source_path, target_path)

    return target_path


def split_dataset(grouped: dict) -> tuple[list[dict], list[dict], list[dict]]:
    random.seed(RANDOM_SEED)

    all_records = []
    train_records = []
    test_records = []

    for label_name, items in grouped.items():
        items = items[:]
        random.shuffle(items)

        n = len(items)
        test_count = decide_split_count(n)

        test_items = items[:test_count]
        train_items = items[test_count:]

        for item in train_items:
            copy_file(item, "train")

            record = {
                "call_id": item["call_id"],
                "doc_type": item["doc_type"],
                "unit": item["unit"],
                "category": item["category"],
                "label_name": item["label_name"],
                "split": "train",
            }

            train_records.append(record)
            all_records.append(record)

        for item in test_items:
            copy_file(item, "test")

            record = {
                "call_id": item["call_id"],
                "doc_type": item["doc_type"],
                "unit": item["unit"],
                "category": item["category"],
                "label_name": item["label_name"],
                "split": "test",
            }

            test_records.append(record)
            all_records.append(record)

    return all_records, train_records, test_records


def save_json(data: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data.sort(
        key=lambda x: (
            x["split"],
            x["doc_type"],
            x["unit"],
            x["category"],
            x["call_id"],
        )
    )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def clear_output_dirs() -> None:
    for path in [TRAIN_DATA_DIR, TEST_DATA_DIR]:
        if path.exists():
            shutil.rmtree(path)


def main():
    clear_output_dirs()

    grouped = collect_files(INPUT_DATA_DIR)

    all_records, train_records, test_records = split_dataset(grouped)

    save_json(all_records, OUTPUT_ALL_JSON)
    save_json(train_records, OUTPUT_TRAIN_JSON)
    save_json(test_records, OUTPUT_TEST_JSON)

    print("切分完成")
    print(f"總類別數：{len(grouped)}")
    print(f"總資料數：{len(all_records)}")
    print(f"Train 數量：{len(train_records)}")
    print(f"Test 數量：{len(test_records)}")
    print(f"輸出資料夾：{TRAIN_DATA_DIR}, {TEST_DATA_DIR}")
    print(f"輸出 JSON：{OUTPUT_ALL_JSON}")


if __name__ == "__main__":
    main()
