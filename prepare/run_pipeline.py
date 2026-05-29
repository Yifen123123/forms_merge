from pathlib import Path
import argparse
import importlib.util
import time
import traceback


PIPELINE = [
    {
        "stage": 0,
        "name": "資料清洗",
        "file": "00_clean_data.py",
    },
    {
        "stage": 1,
        "name": "建立 Label JSON",
        "file": "01_build_label_json.py",
    },
    {
        "stage": 2,
        "name": "類別合併並重建 JSON",
        "file": "02_merge_class_rebuild_json.py",
    },
    {
        "stage": 3,
        "name": "切分 Train/Test",
        "file": "03_split_train_test.py",
    },
    {
        "stage": 4,
        "name": "建立 Category Knowledge Base",
        "file": "04_build_category_knowledge_base.py",
    },
    {
        "stage": 5,
        "name": "建立 Category FAISS Index",
        "file": "05_build_category_faiss_index.py",
    },
    {
        "stage": 6,
        "name": "Two-Stage Category RAG 測試",
        "file": "06_test_two_stage_category_rag.py",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run full classification pipeline."
    )

    parser.add_argument(
        "--start-stage",
        type=int,
        default=0,
        help="從第幾個 stage 開始執行，預設 0",
    )

    parser.add_argument(
        "--end-stage",
        type=int,
        default=6,
        help="執行到第幾個 stage 結束，預設 6",
    )

    parser.add_argument(
        "--list-stages",
        action="store_true",
        help="列出所有 stage，不執行 pipeline",
    )

    return parser.parse_args()


def list_stages():
    print("Pipeline stages:")
    for item in PIPELINE:
        print(f"{item['stage']}: {item['name']} ({item['file']})")


def load_module_from_file(file_path: Path):
    """
    因為檔名是 00_clean_data.py 這種格式，
    不能用一般 import，所以使用 importlib 從檔案路徑載入。
    """
    module_name = file_path.stem

    spec = importlib.util.spec_from_file_location(
        module_name,
        file_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"無法載入 module：{file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def run_stage(stage_info: dict):
    stage = stage_info["stage"]
    name = stage_info["name"]
    file_path = Path(stage_info["file"])

    if not file_path.exists():
        raise FileNotFoundError(f"找不到檔案：{file_path}")

    print("\n" + "=" * 80)
    print(f"Stage {stage}：{name}")
    print(f"File：{file_path}")
    print("=" * 80)

    module = load_module_from_file(file_path)

    if not hasattr(module, "run"):
        raise AttributeError(
            f"{file_path} 裡面找不到 run() 函式。\n"
            f"請確認你已經把 main() 改成 run()。"
        )

    start_time = time.perf_counter()

    module.run()

    elapsed = time.perf_counter() - start_time

    print("-" * 80)
    print(f"Stage {stage} 完成：{name}")
    print(f"耗時：{elapsed:.2f} 秒")


def validate_stage_range(start_stage: int, end_stage: int):
    available_stages = [item["stage"] for item in PIPELINE]

    if start_stage not in available_stages:
        raise ValueError(f"start-stage 不存在：{start_stage}")

    if end_stage not in available_stages:
        raise ValueError(f"end-stage 不存在：{end_stage}")

    if start_stage > end_stage:
        raise ValueError("start-stage 不可以大於 end-stage")


def run_pipeline(start_stage: int, end_stage: int):
    validate_stage_range(start_stage, end_stage)

    selected_stages = [
        item for item in PIPELINE
        if start_stage <= item["stage"] <= end_stage
    ]

    total_start = time.perf_counter()

    print("\n即將執行以下流程：")
    for item in selected_stages:
        print(f"- Stage {item['stage']}：{item['name']}")

    for item in selected_stages:
        run_stage(item)

    total_elapsed = time.perf_counter() - total_start

    print("\n" + "=" * 80)
    print("Pipeline 全部執行完成")
    print(f"執行範圍：Stage {start_stage} → Stage {end_stage}")
    print(f"總耗時：{total_elapsed:.2f} 秒")
    print("=" * 80)


def main():
    args = parse_args()

    if args.list_stages:
        list_stages()
        return

    try:
        run_pipeline(
            start_stage=args.start_stage,
            end_stage=args.end_stage,
        )

    except Exception:
        print("\nPipeline 執行失敗")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
