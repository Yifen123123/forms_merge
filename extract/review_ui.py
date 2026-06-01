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
    path = Path(FORMS_DIR) / f"{call_id}.txt"

    if not path.exists():
        return "找不到原會辦單"

    return path.read_text(encoding="utf-8")


def split_form_text(text: str) -> tuple[str, str]:
    """
    如果原會辦單本身有「問題描述」「需求內容」標題，
    可以簡單切出來。
    如果切不出來，就全部放在問題描述區。
    """

    problem = text
    request = ""

    if "【需求內容】" in text:
        parts = text.split("【需求內容】", 1)
        problem = parts[0].replace("【問題描述】", "").strip()
        request = parts[1].strip()

    elif "需求內容：" in text:
        parts = text.split("需求內容：", 1)
        problem = parts[0].replace("問題描述：", "").strip()
        request = parts[1].strip()

    return problem.strip(), request.strip()


def render_text_block(title: str, content: str):
    st.markdown(f"**{title}**")
    st.info(content if content else "無")


def render_record(row):
    call_id = str(row["call_id"])
    original_text = load_original_form(call_id)

    original_problem, original_request = split_form_text(original_text)

    status = row.get("status", "")

    with st.container(border=True):
        header_left, header_right = st.columns([4, 1])

        with header_left:
            st.subheader(f"Call ID：{call_id}")

        with header_right:
            if status == "success":
                st.success("success")
            elif status:
                st.warning(status)
            else:
                st.info("unknown")

        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("### 原會辦單")

            render_text_block(
                "問題描述",
                original_problem
            )

            render_text_block(
                "需求內容",
                original_request
            )

            with st.expander("查看原會辦單全文"):
                st.text(original_text)

        with col2:
            st.markdown("### LLM 擷取結果")

            render_text_block(
                "問題描述",
                row.get("problem_description", "")
            )

            render_text_block(
                "需求內容",
                row.get("request_content", "")
            )


df = pd.read_csv(
    CSV_FILE,
    dtype={"call_id": str}
)

total_records = len(df)
total_pages = math.ceil(total_records / PAGE_SIZE)

st.title("會辦單擷取結果比對")
st.caption("左側為原會辦單，右側為 LLM 擷取結果。每頁顯示 5 筆資料。")

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
    st.metric(
        "總筆數",
        total_records
    )

start = (page - 1) * PAGE_SIZE
end = start + PAGE_SIZE

page_df = df.iloc[start:end]

st.divider()

for _, row in page_df.iterrows():
    render_record(row)

st.caption(
    f"Showing {start + 1} - {min(end, total_records)} of {total_records} records"
)
