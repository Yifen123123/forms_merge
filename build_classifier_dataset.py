import json
from pathlib import Path
from collections import Counter, defaultdict


MAPPING_PATH = Path("forms_type_mapping.json")
CALLS_DIR = Path("calls")
OUTPUT_DIR = Path("processed")

DATASET_OUTPUT_PATH = OUTPUT_DIR / "classifier_dataset.json"
STATS_OUTPUT_PATH = OUTPUT_DIR / "dataset_stats.json"


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"找不到檔案：{path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_call_id(call_id: str) -> str:
    """
    避免人工建立 JSON 時不小心寫成 xxx.txt
    這裡統一轉成不含副檔名的 call_id
    """
    call_id = str(call_id).strip()

    if call_id.endswith(".txt"):
        call_id = call_id[:-4]

    return call_id


def build_label_name(unit: str, category: str) -> str:
    """
    你的分類標籤格式：
    行政：會辦單位__會辦單類別
    業務：代碼__會辦單類別
    """
    return f"{unit}__{category}"


def convert_mapping_to_classifier_dataset(mapping_data):
    dataset = []

    missing_call_files = []
    duplicated_call_ids = []
    seen_call_ids = set()

    for item_idx, item in enumerate(mapping_data, start=1):
        doc_type = item.get("doc_type", "").strip()
        unit = item.get("unit", "").strip()
        category = item.get("category", "").strip()
        calls = item.get("calls", [])

        if not doc_type:
            raise ValueError(f"第 {item_idx} 筆缺少 doc_type")

        if not unit:
            raise ValueError(f"第 {item_idx} 筆缺少 unit")

        if not category:
            raise ValueError(f"第 {item_idx} 筆缺少 category")

        if not isinstance(calls, list):
            raise TypeError(f"第 {item_idx} 筆的 calls 必須是 list")

        label_name = build_label_name(unit, category)

        for call_idx, call in enumerate(calls, start=1):
            call_id = normalize_call_id(call.get("call_id", ""))

            if not call_id:
                raise ValueError(
                    f"第 {item_idx} 筆，第 {call_idx} 個 calls 缺少 call_id"
                )

            call_file = CALLS_DIR / f"{call_id}.txt"

            if not call_file.exists():
                missing_call_files.append(str(call_file))

            if call_id in seen_call_ids:
                duplicated_call_ids.append(call_id)

            seen_call_ids.add(call_id)

            dataset.append(
                {
                    "call_id": call_id,
                    "doc_type": doc_type,
                    "unit": unit,
                    "category": category,
                    "label_name": label_name
                }
            )

    return dataset, missing_call_files, duplicated_call_ids


def build_statistics(mapping_data, dataset, missing_call_files, duplicated_call_ids):
    total_classes = len(mapping_data)
    total_calls = len(dataset)

    doc_type_counter = Counter()
    label_counter = Counter()
    unit_counter = Counter()

    class_call_count = []

    for item in mapping_data:
        doc_type = item["doc_type"]
        unit = item["unit"]
        category = item["category"]
        calls = item.get("calls", [])

        label_name = build_label_name(unit, category)

        doc_type_counter[doc_type] += 1
        unit_counter[unit] += 1
        label_counter[label_name] += len(calls)

        class_call_count.append(
            {
                "doc_type": doc_type,
                "unit": unit,
                "category": category,
                "label_name": label_name,
                "num_calls": len(calls)
            }
        )

    used_call_ids = {item["call_id"] for item in dataset}

    existing_txt_files = set()
    if CALLS_DIR.exists():
        for path in CALLS_DIR.glob("*.txt"):
            existing_txt_files.add(path.stem)

    unused_txt_files = sorted(existing_txt_files - used_call_ids)

    stats = {
        "summary": {
            "total_classes": total_classes,
            "total_calls": total_calls,
            "num_doc_types": dict(doc_type_counter),
            "num_units_or_codes": len(unit_counter),
            "num_labels": len(label_counter),
            "missing_call_files_count": len(missing_call_files),
            "unused_txt_files_count": len(unused_txt_files),
            "duplicated_call_ids_count": len(duplicated_call_ids)
        },
        "calls_per_label": dict(label_counter),
        "classes": class_call_count,
        "missing_call_files": missing_call_files,
        "unused_txt_files": unused_txt_files,
        "duplicated_call_ids": duplicated_call_ids
    }

    return stats


def main():
    mapping_data = load_json(MAPPING_PATH)

    if not isinstance(mapping_data, list):
        raise TypeError("forms_type_mapping.json 最外層必須是 list")

    dataset, missing_call_files, duplicated_call_ids = (
        convert_mapping_to_classifier_dataset(mapping_data)
    )

    stats = build_statistics(
        mapping_data=mapping_data,
        dataset=dataset,
        missing_call_files=missing_call_files,
        duplicated_call_ids=duplicated_call_ids
    )

    save_json(dataset, DATASET_OUTPUT_PATH)
    save_json(stats, STATS_OUTPUT_PATH)

    print("資料轉換完成")
    print(f"分類器資料：{DATASET_OUTPUT_PATH}")
    print(f"統計資料：{STATS_OUTPUT_PATH}")
    print()
    print("統計摘要：")
    print(json.dumps(stats["summary"], ensure_ascii=False, indent=2))

    if missing_call_files:
        print("\n有 mapping 提到，但 calls/ 找不到的檔案：")
        for path in missing_call_files:
            print(f"- {path}")

    if unused_txt_files:
        print("\n有 txt 檔存在，但沒有被 mapping 使用：")
        for call_id in unused_txt_files:
            print(f"- {call_id}.txt")

    if duplicated_call_ids:
        print("\n有重複使用的 call_id：")
        for call_id in duplicated_call_ids:
            print(f"- {call_id}")


if __name__ == "__main__":
    main()
