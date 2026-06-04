import math
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


def load_original_form(call_id: str) -> str:
    form_path = Path(FORMS_DIR) / f"{call_id}.txt"

    if not form_path.exists():
        return "找不到原會辦單"

    return form_path.read_text(encoding="utf-8")


def clean_value(value) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def render_case(row):
    case_no = clean_value(row.get("case_no", ""))
    problem_description = clean_value(row.get("problem_description", ""))
    request_content = clean_value(row.get("request_content", ""))
    status = clean_value(row.get("status", ""))

    with st.container(border=True):
        if case_no:
            st.markdown(f"#### 會辦單 {case_no}")
        else:
            st.markdown("#### 會辦單")

        if status == "success":
            st.success("success")
        elif status:
            st.warning(status)
        else:
            st.info("unknown")

        st.markdown("**問題描述**")
        st.success(
            problem_description
            if problem_description
            else "無"
        )

        st.markdown("**需求內容**")
        st.success(
            request_content
            if request_content
            else "無"
        )


def render_call_block(call_id: str, group_df: pd.DataFrame):
    original_form = load_original_form(call_id)

    statuses = group_df["status"].dropna().astype(str).unique().tolist()

    with st.container(border=True):
        header_col1, header_col2 = st.columns([4, 1])

        with header_col1:
            st.subheader(f"Call ID：{call_id}")

        with header_col2:
            if all(status == "success" for status in statuses):
                st.success(f"{len(group_df)} 筆")
            else:
                st.warning(" / ".join(statuses) if statuses else "unknown")

        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("### 原會辦單")
            st.info(original_form)

        with col2:
            st.markdown("### LLM 擷取結果")

            for _, row in group_df.iterrows():
                render_case(row)


def main():
    if not Path(CSV_FILE).exists():
        st.error(f"找不到檔案：{CSV_FILE}")
        return

    df = pd.read_csv(
        CSV_FILE,
        dtype={
            "call_id": str,
            "case_no": str
        }
    )

    if df.empty:
        st.warning("extracted_calls.csv 沒有資料")
        return

    df["call_id"] = df["call_id"].astype(str)

    call_ids = df["call_id"].dropna().unique().tolist()

    total_calls = len(call_ids)
    total_pages = math.ceil(total_calls / PAGE_SIZE)

    st.title("會辦單擷取結果比對")
    st.caption(
        "左側為原會辦單全文，右側為 LLM 擷取出的問題描述與需求內容。"
        "一個 Call ID 為一個區塊，每頁顯示 5 個 Call ID。"
    )

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
        st.metric("總通話數", total_calls)

    start_idx = (page - 1) * PAGE_SIZE
    end_idx = start_idx + PAGE_SIZE

    page_call_ids = call_ids[start_idx:end_idx]

    st.divider()

    for call_id in page_call_ids:
        group_df = df[df["call_id"] == call_id].copy()

        group_df["case_no_sort"] = pd.to_numeric(
            group_df["case_no"],
            errors="coerce"
        )

        group_df = group_df.sort_values(
            by="case_no_sort",
            na_position="last"
        )

        render_call_block(call_id, group_df)

    st.caption(
        f"Showing {start_idx + 1} - {min(end_idx, total_calls)} "
        f"of {total_calls} call IDs"
    )


if __name__ == "__main__":
    main()
