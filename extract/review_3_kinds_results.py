import math
from pathlib import Path

import pandas as pd
import streamlit as st


# =========================
# 基本設定
# =========================

FORMS_DIR = "data/forms"
PAGE_SIZE = 5

MODEL_RESULT_FILES = {
    "GPT-OSS 20B": "results_gpt-oss_20b/extracted_calls.csv",
    "Qwen3 8B": "results_qwen3_8b/extracted_calls.csv",
    "Qwen2.5 14B": "results_qwen2.5_14b/extracted_calls.csv",
}


st.set_page_config(
    page_title="多模型會辦單擷取比對",
    layout="wide"
)


# =========================
# 工具函數
# =========================

def clean_value(value) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def load_original_form(call_id: str) -> str:
    path = Path(FORMS_DIR) / f"{call_id}.txt"

    if not path.exists():
        return "找不到原會辦單"

    return path.read_text(encoding="utf-8")


def load_model_results() -> dict[str, pd.DataFrame]:
    model_dfs = {}

    for model_name, file_path in MODEL_RESULT_FILES.items():
        path = Path(file_path)

        if not path.exists():
            st.error(f"找不到模型結果檔：{file_path}")
            continue

        df = pd.read_csv(path, dtype={"call_id": str})

        required_cols = {
            "call_id",
            "problem_description",
            "request_content"
        }

        missing_cols = required_cols - set(df.columns)

        if missing_cols:
            st.error(
                f"{file_path} 缺少欄位：{missing_cols}"
            )
            continue

        df = df.set_index("call_id")

        model_dfs[model_name] = df

    return model_dfs


def get_all_call_ids(model_dfs: dict[str, pd.DataFrame]) -> list[str]:
    call_ids = set()

    for df in model_dfs.values():
        call_ids.update(df.index.astype(str).tolist())

    return sorted(call_ids)


def render_model_result(model_name: str, row_data):
    st.markdown(f"### {model_name}")

    if row_data is None:
        st.warning("此模型沒有這筆資料")
        return

    problem_description = clean_value(
        row_data.get("problem_description", "")
    )

    request_content = clean_value(
        row_data.get("request_content", "")
    )

    status = clean_value(
        row_data.get("status", "")
    )

    if status == "success":
        st.success("status：success")
    elif status:
        st.warning(f"status：{status}")
    else:
        st.info("status：unknown")

    st.markdown("**問題描述**")
    st.info(
        problem_description
        if problem_description
        else "無"
    )

    st.markdown("**需求內容**")
    st.info(
        request_content
        if request_content
        else "無"
    )


def render_record(call_id: str, model_dfs: dict[str, pd.DataFrame]):
    original_form = load_original_form(call_id)

    with st.container(border=True):
        st.subheader(f"Call ID：{call_id}")

        columns = st.columns(4, gap="medium")

        with columns[0]:
            st.markdown("### 原會辦單")
            st.info(original_form)

        for col, (model_name, df) in zip(
            columns[1:],
            model_dfs.items()
        ):
            with col:
                if call_id in df.index:
                    row_data = df.loc[call_id]
                    render_model_result(model_name, row_data)
                else:
                    render_model_result(model_name, None)


# =========================
# Main
# =========================

def main():
    st.title("多模型會辦單擷取結果比對")
    st.caption(
        "左側為原會辦單，右側依序顯示 GPT-OSS 20B、Qwen3 8B、Qwen2.5 14B 的擷取結果。"
    )

    model_dfs = load_model_results()

    if not model_dfs:
        st.stop()

    call_ids = get_all_call_ids(model_dfs)

    if not call_ids:
        st.warning("沒有可顯示的 call_id")
        st.stop()

    total_records = len(call_ids)
    total_pages = math.ceil(total_records / PAGE_SIZE)

    top_col1, top_col2, top_col3 = st.columns([1, 1, 4])

    with top_col1:
        page = st.number_input(
            "目前頁數",
            min_value=1,
            max_value=total_pages,
            value=1,
            step=1
        )

    with top_col2:
        st.metric("總筆數", total_records)

    start_idx = (page - 1) * PAGE_SIZE
    end_idx = start_idx + PAGE_SIZE

    page_call_ids = call_ids[start_idx:end_idx]

    st.divider()

    for call_id in page_call_ids:
        render_record(call_id, model_dfs)

    st.caption(
        f"Showing {start_idx + 1} - {min(end_idx, total_records)} "
        f"of {total_records} records"
    )


if __name__ == "__main__":
    main()
