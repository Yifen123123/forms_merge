import math
import html
from pathlib import Path

import pandas as pd
import streamlit as st


CSV_FILE = "results/extracted_calls.csv"
FORMS_DIR = "data/forms"
PAGE_SIZE = 5


st.set_page_config(
    page_title="會辦單擷取比對",
    layout="wide"
)


def safe_text(value):
    if pd.isna(value):
        return ""
    return html.escape(str(value))


def load_original_form(call_id: str) -> str:
    path = Path(FORMS_DIR) / f"{call_id}.txt"

    if not path.exists():
        return "找不到原會辦單"

    return path.read_text(encoding="utf-8")


def render_card(call_id, original_text, problem_description, request_content, status):
    original_text = safe_text(original_text)
    problem_description = safe_text(problem_description)
    request_content = safe_text(request_content)
    status = safe_text(status)

    html_block = f"""
    <div class="record-card">
        <div class="left-panel">
            <div class="panel-title">原會辦單內容｜ORIGINAL MEMO</div>
            <div class="call-badge">{call_id}</div>

            <div class="memo-box">
                <pre>{original_text}</pre>
            </div>
        </div>

        <div class="right-panel">
            <div class="panel-title">LLM 擷取結果｜AI EXTRACTION</div>

            <div class="confidence">
                🤖 擷取狀態：{status}
            </div>

            <div class="field-row">
                <div class="field-label">問題描述</div>
                <div class="ai-box">{problem_description}</div>
            </div>

            <div class="field-row">
                <div class="field-label">需求內容</div>
                <div class="ai-box">{request_content}</div>
            </div>
        </div>
    </div>
    """

    st.markdown(html_block, unsafe_allow_html=True)


st.markdown(
    """
    <style>
    .main {
        background-color: #f5f7fa;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    .page-title {
        font-size: 28px;
        font-weight: 800;
        margin-bottom: 6px;
        color: #1f2937;
    }

    .page-subtitle {
        font-size: 14px;
        color: #6b7280;
        margin-bottom: 24px;
    }

    .record-card {
        display: grid;
        grid-template-columns: 1fr 1fr;
        border: 1px solid #d1d5db;
        background-color: white;
        margin-bottom: 0px;
        min-height: 245px;
    }

    .left-panel,
    .right-panel {
        padding: 20px 24px;
    }

    .left-panel {
        border-right: 1px solid #d1d5db;
        background-color: #f9fafb;
    }

    .right-panel {
        background-color: #f8fbfb;
    }

    .panel-title {
        font-size: 12px;
        font-weight: 700;
        color: #374151;
        margin-bottom: 18px;
        letter-spacing: 0.3px;
    }

    .call-badge {
        display: inline-block;
        background-color: #1f2937;
        color: white;
        font-size: 12px;
        font-weight: 700;
        padding: 5px 10px;
        border-radius: 2px;
        margin-bottom: 12px;
    }

    .memo-box {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 2px;
        padding: 12px 14px;
        font-size: 14px;
        line-height: 1.7;
        color: #111827;
        min-height: 130px;
    }

    .memo-box pre {
        white-space: pre-wrap;
        word-break: break-word;
        margin: 0;
        font-family: inherit;
    }

    .confidence {
        color: #047857;
        font-weight: 800;
        font-size: 14px;
        margin-bottom: 22px;
    }

    .field-row {
        display: grid;
        grid-template-columns: 90px 1fr;
        align-items: start;
        margin-bottom: 14px;
    }

    .field-label {
        font-size: 13px;
        color: #374151;
        padding-top: 7px;
    }

    .ai-box {
        background-color: #dff7f4;
        border: 1px solid #99d8d0;
        color: #111827;
        padding: 8px 12px;
        border-radius: 2px;
        font-size: 14px;
        line-height: 1.6;
        min-height: 38px;
    }

    .pagination-info {
        font-size: 13px;
        color: #4b5563;
        margin-top: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


df = pd.read_csv(CSV_FILE, dtype={"call_id": str})

total_records = len(df)
total_pages = math.ceil(total_records / PAGE_SIZE)

st.markdown(
    """
    <div class="page-title">會辦單擷取結果比對</div>
    <div class="page-subtitle">一次顯示 5 筆，左側為原會辦單，右側為 LLM 擷取結果。</div>
    """,
    unsafe_allow_html=True
)

col_page, col_jump = st.columns([1, 4])

with col_page:
    page = st.number_input(
        "頁數",
        min_value=1,
        max_value=total_pages,
        value=1,
        step=1
    )

start = (page - 1) * PAGE_SIZE
end = start + PAGE_SIZE
page_df = df.iloc[start:end]

for _, row in page_df.iterrows():
    call_id = str(row["call_id"])
    original_text = load_original_form(call_id)

    render_card(
        call_id=call_id,
        original_text=original_text,
        problem_description=row.get("problem_description", ""),
        request_content=row.get("request_content", ""),
        status=row.get("status", "success")
    )

st.markdown(
    f"""
    <div class="pagination-info">
        Showing {start + 1} - {min(end, total_records)} of {total_records} records
    </div>
    """,
    unsafe_allow_html=True
)
