from pathlib import Path
import argparse
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# 直接從原本測試程式 import 可重用函式
from two_stage_rag_test import (
    load_json,
    save_json,
    call_llm_json,
    build_unit_documents,
    build_unit_faiss,
    retrieve_units,
    retrieve_categories_within_unit,
    build_unit_prompt,
    build_category_prompt,
    validate_unit_prediction,
    validate_category_prediction,
)


EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"

CATEGORY_INDEX_PATH = Path("processed/category_faiss/category_rules.index")
CATEGORY_METADATA_PATH = Path("processed/category_faiss/category_rules_metadata.json")

TOP_K_UNIT = 5
TOP_K_CATEGORY = 5
MAX_CALL_CHARS = 5000


def read_call_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"找不到通話檔案：{path}")

    text = path.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError(f"通話檔案是空的：{path}")

    return text[:MAX_CALL_CHARS]


def predict_single_call(call_text: str) -> dict:
    print("載入 embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("載入 category FAISS...")
    category_index = faiss.read_index(str(CATEGORY_INDEX_PATH))
    category_metadata = load_json(CATEGORY_METADATA_PATH)

    print("建立 unit-level FAISS...")
    unit_docs = build_unit_documents(category_metadata)
    unit_index, _ = build_unit_faiss(unit_docs, embedding_model)

    # Stage 1: Unit
    unit_candidates = retrieve_units(
        call_text=call_text,
        model=embedding_model,
        unit_index=unit_index,
        unit_docs=unit_docs,
        top_k=TOP_K_UNIT,
    )

    if not unit_candidates:
        raise ValueError("Stage 1 沒有取回任何 unit candidates")

    unit_prompt = build_unit_prompt(call_text, unit_candidates)

    unit_raw = call_llm_json(
        unit_prompt,
        retry_hint='{"pred_candidate_index": 0, "confidence": 0.0, "reason": "", "evidence_from_call": [], "need_human_review": true}',
    )

    unit_pred = validate_unit_prediction(unit_raw, unit_candidates)

    # Stage 2: Category
    category_candidates = retrieve_categories_within_unit(
        call_text=call_text,
        pred_doc_type=unit_pred["doc_type"],
        pred_unit=unit_pred["unit"],
        model=embedding_model,
        category_index=category_index,
        category_metadata=category_metadata,
        top_k=TOP_K_CATEGORY,
    )

    if not category_candidates:
        raise ValueError(
            f"Stage 2 沒有取回任何 category candidates："
            f"{unit_pred['doc_type']} / {unit_pred['unit']}"
        )

    category_prompt = build_category_prompt(call_text, category_candidates)

    category_raw = call_llm_json(
        category_prompt,
        retry_hint='{"pred_candidate_index": 0, "confidence": 0.0, "reason": "", "evidence_from_call": [], "possible_alternatives": [], "need_human_review": true}',
    )

    final_pred = validate_category_prediction(category_raw, category_candidates)

    return {
        "unit_prediction": unit_pred,
        "final_prediction": final_pred,
        "unit_candidates": [
            {
                "candidate_index": c["candidate_index"],
                "rank": c["rank"],
                "retrieval_score": c["retrieval_score"],
                "doc_type": c["doc_type"],
                "unit": c["unit"],
            }
            for c in unit_candidates
        ],
        "category_candidates": [
            {
                "candidate_index": c["candidate_index"],
                "rank": c["rank"],
                "retrieval_score": c["retrieval_score"],
                "label_name": c["label_name"],
                "doc_type": c["doc_type"],
                "unit": c["unit"],
                "category": c["category"],
            }
            for c in category_candidates
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="要分類的通話 .txt 檔案路徑",
    )
    parser.add_argument(
        "--output",
        default="processed/single_call_prediction.json",
        help="預測結果輸出路徑",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    call_text = read_call_file(input_path)

    result = predict_single_call(call_text)

    save_json(result, output_path)

    print("=" * 80)
    print("單筆通話分類完成")
    print(f"預測 doc_type：{result['final_prediction']['doc_type']}")
    print(f"預測 unit：{result['final_prediction']['unit']}")
    print(f"預測 category：{result['final_prediction']['category']}")
    print(f"預測 label_name：{result['final_prediction']['label_name']}")
    print(f"confidence：{result['final_prediction']['confidence']}")
    print(f"need_human_review：{result['final_prediction']['need_human_review']}")
    print(f"結果輸出：{output_path}")


if __name__ == "__main__":
    main()
