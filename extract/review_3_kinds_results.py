import math
import html
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

BOX_COLORS = {
    "原會辦單": {
        "bg": "#E5E7EB",
        "border": "#6B7280",
    },
    "GPT-OSS 20B": {
        "bg": "#BFDBFE",
        "border": "#2563EB",
    },
    "Qwen3 8B": {
        "bg": "#BBF7D0",
        "border": "#16A34A",
    },
    "Qwen2.5 14B": {
        "bg": "#FED7AA",
        "border": "#EA580C",
    },
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
    return str(value).strip()


def remove_html_residue(text: str) -> str:
    """
    清除模型輸出或 CSV 裡可能殘留的 HTML tag。
    避免 Streamlit 畫面底部出現 </div>。
    """
    residues = [
        "<div>",
        "</div>",
        "<br>",
        "<br/>",
        "<br />",
    ]

    for tag in residues:
        text = text.replace(tag, "")

    return text


def html_text(value) -> str:
    text = clean_value(value)
    text = remove_html_residue(text)
    text = html.escape(text)

    return text.replace("\n", "<br>")


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

        df = pd.read_csv(
            path,
            dtype={"call_id": str}
        )

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
        call_ids.update(df.index.tolist())

    forms_dir = Path(FORMS_DIR)

    if forms_dir.exists():
        call_ids.update(
            path.stem
            for path in forms_dir.glob("*.txt")
        )

    return sorted(call_ids)


# =========================
# UI 元件
# =========================

def render_color_box(
    content: str,
    style_key: str,
    min_height: int = 260
):
    color = BOX_COLORS[style_key]

    st.markdown(
        f"""
        <div style="
            background-color: {color['bg']};
            border: 1px solid {color['border']};
            border-radius: 8px;
            padding: 14px;
            min-height: {min_height}px;
            line-height: 1.7;
            font-size: 14px;
            color: #111827;
            word-break: break-word;
            overflow-wrap: break-word;
        ">
            {content}
        </div>
        """,
        unsafe_allow_html=True
    )


def render_model_column(
    model_name: str,
    row_data
):
    st.markdown(f"### {model_name}")

    if row_data is None:
        render_color_box(
            content="此模型沒有這筆資料",
            style_key=model_name,
            min_height=260
        )
        return

    problem = html_text(
        row_data.get("problem_description", "")
    )

    request = html_text(
        row_data.get("request_content", "")
    )

    if not problem:
        problem = "無"

    if not request:
        request = "無"

    content = f"""
        <strong>問題描述</strong><br>
        {problem}

        <br><br>

        <strong>需求內容</strong><br>
        {request}
    """

    render_color_box(
        content=content,
        style_key=model_name,
        min_height=260
    )


def render_record(
    call_id: str,
    model_dfs: dict[str, pd.DataFrame]
):
    original_form = html_text(
        load_original_form(call_id)
    )

    with st.container(border=True):
        st.subheader(f"Call ID：{call_id}")

        col1, col2, col3, col4 = st.columns(
            4,
            gap="medium"
        )

        with col1:
            st.markdown("### 原會辦單")
            render_color_box(
                content=original_form,
                style_key="原會辦單",
                min_height=260
            )

        with col2:
            model_name = "GPT-OSS 20B"
            df = model_dfs.get(model_name)

            row_data = (
                df.loc[call_id]
                if df is not None
                and call_id in df.index
                else None
            )

            render_model_column(
                model_name=model_name,
                row_data=row_data
            )

        with col3:
            model_name = "Qwen3 8B"
            df = model_dfs.get(model_name)

            row_data = (
                df.loc[call_id]
                if df is not None
                and call_id in df.index
                else None
            )

            render_model_column(
                model_name=model_name,
                row_data=row_data
            )

        with col4:
            model_name = "Qwen2.5 14B"
            df = model_dfs.get(model_name)

            row_data = (
                df.loc[call_id]
                if df is not None
                and call_id in df.index
                else None
            )

            render_model_column(
                model_name=model_name,
                row_data=row_data
            )


# =========================
# Main
# =========================

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

    col_page, col_total, _ = st.columns(
        [1, 1, 4]
    )

    with col_page:
        page = st.number_input(
            "目前頁數",
            min_value=1,
            max_value=total_pages,
            value=1,
            step=1
        )

    with col_total:
        st.metric(
            "總筆數",
            total_records
        )

    start_idx = (page - 1) * PAGE_SIZE
    end_idx = start_idx + PAGE_SIZE

    st.divider()

    for call_id in call_ids[start_idx:end_idx]:
        render_record(
            call_id=call_id,
            model_dfs=model_dfs
        )

    st.caption(
        f"Showing {start_idx + 1} - {min(end_idx, total_records)} "
        f"of {total_records} records"
    )


if __name__ == "__main__":
    main()
