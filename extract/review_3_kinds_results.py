import math
import html
from pathlib import Path

import pandas as pd
import streamlit as st


FORMS_DIR = "data/forms"
PAGE_SIZE = 5

MODEL_RESULT_FILES = {
    "GPT-OSS 20B": "results_gpt-oss_20b/extracted_calls.csv",
    "Qwen3 8B": "results_qwen3_8b/extracted_calls.csv",
    "Qwen2.5 14B": "results_qwen2.5_14b/extracted_calls.csv",
}

BOX_STYLES = {
    "原會辦單": {
        "bg": "#F8FAFC",
        "border": "#64748B",
    },
    "GPT-OSS 20B": {
        "bg": "#EFF6FF",
        "border": "#2563EB",
    },
    "Qwen3 8B": {
        "bg": "#ECFDF5",
        "border": "#10B981",
    },
    "Qwen2.5 14B": {
        "bg": "#FFF7ED",
        "border": "#F97316",
    },
}


st.set_page_config(
    page_title="多模型會辦單擷取比對",
    layout="wide"
)


def clean_value(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def escape_text(value) -> str:
    return html.escape(clean_value(value))


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

        required_columns = {
            "call_id",
            "problem_description",
            "request_content",
        }

        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            st.error(
                f"{file_path} 缺少欄位：{', '.join(missing_columns)}"
            )
            continue

        df["call_id"] = df["call_id"].astype(str)
        df = df.set_index("call_id")

        model_dfs[model_name] = df

    return model_dfs


def get_all_call_ids(model_dfs: dict[str, pd.DataFrame]) -> list[str]:
    call_ids = set()

    for df in model_dfs.values():
        call_ids.update(df.index.astype(str).tolist())

    forms_path = Path(FORMS_DIR)
    if forms_path.exists():
        call_ids.update(
            path.stem
            for path in forms_path.glob("*.txt")
        )

    return sorted(call_ids)


def colored_box(title: str, content: str, style_key: str):
    style = BOX_STYLES[style_key]

    st.markdown(
        f"""
        <div style="
            background-color:{style['bg']};
            border-left:6px solid {style['border']};
            border-radius:10px;
            padding:16px;
            min-height:360px;
            white-space:pre-wrap;
            line-height:1.7;
            font-size:14px;
            word-break:break-word;
            overflow-wrap:break-word;
        ">
            <div style="
                font-size:18px;
                font-weight:800;
                margin-bottom:14px;
                color:#111827;
            ">
                {html.escape(title)}
            </div>
            {content}
        </div>
        """,
        unsafe_allow_html=True
    )


def build_model_content(row_data) -> str:
    if row_data is None:
        return "<span style='color:#9CA3AF;'>此模型沒有這筆資料</span>"

    problem_description = escape_text(
        row_data.get("problem_description", "")
    )

    request_content = escape_text(
        row_data.get("request_content", "")
    )

    if not problem_description:
        problem_description = "<span style='color:#9CA3AF;'>無</span>"

    if not request_content:
        request_content = "<span style='color:#9CA3AF;'>無</span>"

    return f"""
<b>問題描述</b>
<div style="margin-top:6px; margin-bottom:18px;">
{problem_description}
</div>

<b>需求內容</b>
<div style="margin-top:6px;">
{request_content}
</div>
"""


def render_record(call_id: str, model_dfs: dict[str, pd.DataFrame]):
    original_form = escape_text(
        load_original_form(call_id)
    )

    with st.container(border=True):
        st.subheader(f"Call ID：{call_id}")

        col1, col2, col3, col4 = st.columns(
            4,
            gap="medium"
        )

        with col1:
            colored_box(
                title="原會辦單",
                content=original_form,
                style_key="原會辦單"
            )

        with col2:
            model_name = "GPT-OSS 20B"
            df = model_dfs.get(model_name)

            row_data = (
                df.loc[call_id]
                if df is not None and call_id in df.index
                else None
            )

            colored_box(
                title=model_name,
                content=build_model_content(row_data),
                style_key=model_name
            )

        with col3:
            model_name = "Qwen3 8B"
            df = model_dfs.get(model_name)

            row_data = (
                df.loc[call_id]
                if df is not None and call_id in df.index
                else None
            )

            colored_box(
                title=model_name,
                content=build_model_content(row_data),
                style_key=model_name
            )

        with col4:
            model_name = "Qwen2.5 14B"
            df = model_dfs.get(model_name)

            row_data = (
                df.loc[call_id]
                if df is not None and call_id in df.index
                else None
            )

            colored_box(
                title=model_name,
                content=build_model_content(row_data),
                style_key=model_name
            )


def main():
    st.title("多模型會辦單擷取結果比對")
    st.caption(
        "由左至右：原會辦單、GPT-OSS 20B、Qwen3 8B、Qwen2.5 14B。每頁顯示 5 筆。"
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

    top_col1, top_col2, _ = st.columns([1, 1, 4])

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
