from pathlib import Path

import pandas as pd


FORMS_DIR = "data/forms"

MODEL_FILES = {
    "gpt-oss:20b 擷取結果": "results_gpt-oss_20b/extracted_calls.csv",
    "qwen2.5:14b 擷取結果": "results_qwen2.5_14b/extracted_calls.csv",
    "qwen3:8b 擷取結果": "results_qwen3_8b/extracted_calls.csv",
}

OUTPUT_FILE = "results/model_comparison.xlsx"


def clean_value(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def load_original_form(call_id: str) -> str:
    path = Path(FORMS_DIR) / f"{call_id}.txt"

    if not path.exists():
        return ""

    return path.read_text(encoding="utf-8").strip()


def load_model_csv(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, dtype={"call_id": str})

    required_cols = {
        "call_id",
        "problem_description",
        "request_content",
    }

    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(
            f"{file_path} 缺少欄位：{missing}"
        )

    df["model_result"] = (
        "問題描述：\n"
        + df["problem_description"].fillna("").astype(str)
        + "\n\n需求內容：\n"
        + df["request_content"].fillna("").astype(str)
    )

    return df[["call_id", "model_result"]]


def main():
    output_rows = []

    model_dfs = {}

    for model_col_name, file_path in MODEL_FILES.items():
        model_df = load_model_csv(file_path)
        model_df = model_df.set_index("call_id")
        model_dfs[model_col_name] = model_df

    call_ids = set()

    for df in model_dfs.values():
        call_ids.update(df.index.tolist())

    forms_path = Path(FORMS_DIR)

    if forms_path.exists():
        call_ids.update(
            path.stem for path in forms_path.glob("*.txt")
        )

    for call_id in sorted(call_ids):
        row = {
            "call_id": call_id,
            "原會辦單": load_original_form(call_id),
        }

        for model_col_name, df in model_dfs.items():
            if call_id in df.index:
                row[model_col_name] = clean_value(
                    df.loc[call_id, "model_result"]
                )
            else:
                row[model_col_name] = ""

        output_rows.append(row)

    result_df = pd.DataFrame(output_rows)

    Path("results").mkdir(exist_ok=True)

    with pd.ExcelWriter(
        OUTPUT_FILE,
        engine="openpyxl"
    ) as writer:
        result_df.to_excel(
            writer,
            index=False,
            sheet_name="模型比較"
        )

        worksheet = writer.sheets["模型比較"]

        worksheet.column_dimensions["A"].width = 20
        worksheet.column_dimensions["B"].width = 55
        worksheet.column_dimensions["C"].width = 55
        worksheet.column_dimensions["D"].width = 55
        worksheet.column_dimensions["E"].width = 55

        for row in worksheet.iter_rows():
            for cell in row:
                cell.alignment = cell.alignment.copy(
                    wrap_text=True,
                    vertical="top"
                )

    print(f"已輸出：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
