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

MODEL_COLORS = {
    "原會辦單": {
        "bg": "#F8FAFC",
        "border": "#64748B",
        "title": "#334155",
    },
    "GPT-OSS 20B": {
        "bg": "#EFF6FF",
        "border": "#3B82F6",
        "title": "#1D4ED8",
    },
    "Qwen3 8B": {
        "bg": "#ECFDF5",
        "border": "#10B981",
        "title": "#047857",
    },
    "Qwen2.5 14B": {
        "bg": "#FFF7ED",
        "border": "#F97316",
        "title": "#C2410C",
    },
}


st.set_page_config(
    page_title="多模型會辦單擷取比對",
    layout="wide"
)


# =========================
# CSS
# =========================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    .main-title {
        font-size: 30px;
        font-weight: 800;
        color: #111827;
        margin-bottom: 4px;
    }

    .sub-title {
        font-size: 14px;
        color: #6B7280;
        margin-bottom: 20px;
    }

    .record-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 14px;
        padding: 20px 22px;
        margin-bottom: 28px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }

    .call-id {
        font-size: 20px;
        font-weight: 800;
        color: #111827;
        margin-bottom: 16px;
    }

    .compare-grid {
        display: grid;
        grid-template-columns: 1.25fr 1fr 1fr 1fr;
        gap: 14px;
        align-items: stretch;
    }

    .compare-box {
        border-left: 7px solid;
        border-radius: 12px;
        padding: 16px 16px;
        min-height: 260px;
        box-sizing: border-box;
        overflow-wrap: break-word;
        word-break: break-word;
    }

    .box-title {
        font-size: 17px;
        font-weight: 800;
        margin-bottom: 14px;
    }

    .field-title {
        font-size: 14px;
        font-weight: 800;
        color: #374151;
        margin-top: 12px;
        margin-bottom: 6px;
    }

    .field-content {
        font-size: 14px;
        line-height: 1.75;
        color: #111827;
        white-space: pre-wrap;
    }

    .original-content {
        font-size: 14px;
        line-height: 1.75;
        color: #111827;
        white-space: pre-wrap;
    }

    .missing-text {
        color: #9CA3AF;
        font-style: italic;
    }

    @media (max-width: 1200px) {
        .compare-grid {
            grid-template-columns: 1fr 1fr;
        }
    }

    @media (max-width: 760px) {
        .compare-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# 工具函數
# =========================

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

    form_ids = [
        path.stem
        for path in Path(FORMS_DIR).glob("*.txt")
    ]

    call_ids.update(form_ids)

    return sorted(call_ids)


def render_original_box(original_form: str) -> str:
    color = MODEL_COLORS["原會辦單"]

    content = escape_text(original_form)

    return f"""
    <div class="compare-box"
         style="
            background-color:{color['bg']};
            border-left-color:{color['border']};
         ">
        <div class="box-title" style="color:{color['title']};">
            原會辦單
        </div>
        <div class="original-content">{content}</div>
    </div>
    """


def render_model_box(model_name: str, row_data) -> str:
    color = MODEL_COLORS[model_name]

    if row_data is None:
        problem_description = '<span class="missing-text">此模型沒有這筆資料</span>'
        request_content = '<span class="missing-text">此模型沒有這筆資料</span>'
    else:
        problem_description_text = escape_text(
            row_data.get("problem_description", "")
        )

        request_content_text = escape_text(
            row_data.get("request_content", "")
        )

        problem_description = (
            problem_description_text
            if problem_description_text
            else '<span class="missing-text">無</span>'
        )

        request_content = (
            request_content_text
            if request_content_text
            else '<span class="missing-text">無</span>'
        )

    return f"""
    <div class="compare-box"
         style="
            background-color:{color['bg']};
            border-left-color:{color['border']};
         ">
        <div class="box-title" style="color:{color['title']};">
            {model_name}
        </div>

        <div class="field-title">問題描述</div>
        <div class="field-content">{problem_description}</div>

        <div class="field-title">需求內容</div>
        <div class="field-content">{request_content}</div>
    </div>
    """


def render_record(call_id: str, model_dfs: dict[str, pd.DataFrame]) -> None:
    original_form = load_original_form(call_id)

    boxes = []
    boxes.append(render_original_box(original_form))

    for model_name, df in model_dfs.items():
        if call_id in df.index:
            row_data = df.loc[call_id]
        else:
            row_data = None

        boxes.append(
            render_model_box(model_name, row_data)
        )

    html_block = f"""
    <div class="record-card">
        <div class="call-id">Call ID：{html.escape(call_id)}</div>
        <div class="compare-grid">
            {''.join(boxes)}
        </div>
    </div>
    """

    st.markdown(html_block, unsafe_allow_html=True)


# =========================
# Main
# =========================

def main():
    st.markdown(
        """
        <div class="main-title">多模型會辦單擷取結果比對</div>
        <div class="sub-title">
            同時比較原會辦單、GPT-OSS 20B、Qwen3 8B、Qwen2.5 14B 的擷取結果。
        </div>
        """,
        unsafe_allow_html=True
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
