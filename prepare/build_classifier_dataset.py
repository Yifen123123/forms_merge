from pathlib import Path
import json


DATA_DIR = Path("data")
OUTPUT_PATH = Path("classifier_dataset.json")


def build_dataset(data_dir: Path) -> list[dict]:
    records = []

    for txt_file in data_dir.rglob("*.txt"):
        relative_parts = txt_file.relative_to(data_dir).parts

        if len(relative_parts) != 4:
            print(f"略過格式不符檔案：{txt_file}")
            continue

        doc_type, unit, category, filename = relative_parts

        call_id = Path(filename).stem

        record = {
            "call_id": call_id,
            "doc_type": doc_type,
            "unit": unit,
            "category": category,
            "label_name": f"{unit}_{category}",
        }

        records.append(record)

    records.sort(
        key=lambda x: (
            x["doc_type"],
            x["unit"],
            x["category"],
            x["call_id"],
        )
    )

    return records


def save_json(data: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=4
        )


def main():
    records = build_dataset(DATA_DIR)
    save_json(records, OUTPUT_PATH)

    print(f"完成輸出：{OUTPUT_PATH}")
    print(f"總筆數：{len(records)}")


if __name__ == "__main__":
    main()
