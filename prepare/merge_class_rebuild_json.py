from pathlib import Path
import csv
import json
import shutil


INPUT_DATA_DIR = Path("data")
OUTPUT_DATA_DIR = Path("data_merged")
MERGE_RULES_PATH = Path("merge_rules.csv")
OUTPUT_JSON_PATH = Path("processed/classifier_dataset.json")


def load_merge_rules(csv_path: Path) -> dict:
    """
    回傳格式：
    {
        ("行政會辦單", "會辦單位B", "類別A"): ("會辦單位A", "類別B")
    }
    """
    rules = {}

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        required_columns = {
            "doc_type",
            "old_unit",
            "old_category",
            "new_unit",
            "new_category",
        }

        if not required_columns.issubset(reader.fieldnames):
            raise ValueError(f"CSV 必須包含欄位：{required_columns}")

        for row in reader:
            doc_type = row["doc_type"].strip()
            old_unit = row["old_unit"].strip()
            old_category = row["old_category"].strip()
            new_unit = row["new_unit"].strip()
            new_category = row["new_category"].strip()

            if not all([
                doc_type,
                old_unit,
                old_category,
                new_unit,
                new_category,
            ]):
                continue

            rules[(doc_type, old_unit, old_category)] = (
                new_unit,
                new_category,
            )

    return rules


def get_merged_label(
    doc_type: str,
    unit: str,
    category: str,
    merge_rules: dict
) -> tuple[str, str]:
    return merge_rules.get(
        (doc_type, unit, category),
        (unit, category)
    )


def merge_data_folders(
    input_dir: Path,
    output_dir: Path,
    merge_rules: dict
) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)

    copied_count = 0

    for txt_file in input_dir.rglob("*.txt"):
        relative_parts = txt_file.relative_to(input_dir).parts

        if len(relative_parts) != 4:
            print(f"略過格式不符檔案：{txt_file}")
            continue

        doc_type, old_unit, old_category, filename = relative_parts

        new_unit, new_category = get_merged_label(
            doc_type,
            old_unit,
            old_category,
            merge_rules
        )

        target_path = (
            output_dir
            / doc_type
            / new_unit
            / new_category
            / filename
        )

        target_path.parent.mkdir(parents=True, exist_ok=True)

        if target_path.exists():
            raise FileExistsError(
                f"檔名重複，無法合併：{target_path}"
            )

        shutil.copy2(txt_file, target_path)
        copied_count += 1

    print(f"資料夾合併完成，共複製 {copied_count} 份 txt 檔案")
    print(f"輸出資料夾：{output_dir}")


def build_classifier_dataset(data_dir: Path) -> list[dict]:
    records = []

    for txt_file in data_dir.rglob("*.txt"):
        relative_parts = txt_file.relative_to(data_dir).parts

        if len(relative_parts) != 4:
            print(f"略過格式不符檔案：{txt_file}")
            continue

        doc_type, unit, category, filename = relative_parts
        call_id = Path(filename).stem

        records.append({
            "call_id": call_id,
            "doc_type": doc_type,
            "unit": unit,
            "category": category,
            "label_name": f"{unit}_{category}",
        })

    records.sort(
        key=lambda x: (
            x["doc_type"],
            x["unit"],
            x["category"],
            x["call_id"],
        )
    )

    return records


def save_json(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            records,
            f,
            ensure_ascii=False,
            indent=4
        )


def main():
    merge_rules = load_merge_rules(MERGE_RULES_PATH)

    print(f"讀取合併規則：{len(merge_rules)} 筆")

    merge_data_folders(
        input_dir=INPUT_DATA_DIR,
        output_dir=OUTPUT_DATA_DIR,
        merge_rules=merge_rules
    )

    records = build_classifier_dataset(OUTPUT_DATA_DIR)
    save_json(records, OUTPUT_JSON_PATH)

    print(f"JSON 輸出完成：{OUTPUT_JSON_PATH}")
    print(f"總資料筆數：{len(records)}")


if __name__ == "__main__":
    main()
