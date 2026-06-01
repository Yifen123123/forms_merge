import math
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="會辦單比對系統",
    layout="wide"
)

# =====================
# 載入資料
# =====================

CSV_FILE = "results/extracted_calls.csv"
FORMS_DIR = "data/forms"

df = pd.read_csv(CSV_FILE)

# =====================
# Sidebar
# =====================

st.sidebar.title("會辦單比對")

records_per_page = 5

total_pages = math.ceil(
    len(df) / records_per_page
)

page = st.sidebar.selectbox(
    "頁數",
    range(1, total_pages + 1)
)

# =====================
# 分頁
# =====================

start_idx = (
    (page - 1)
    * records_per_page
)

end_idx = start_idx + records_per_page

page_df = df.iloc[
    start_idx:end_idx
]

# =====================
# Header
# =====================

st.title("📄 客服通話擷取結果比對")

st.caption(
    f"共 {len(df)} 筆資料 ｜ "
    f"第 {page}/{total_pages} 頁"
)

# =====================
# 顯示資料
# =====================

for _, row in page_df.iterrows():

    call_id = str(row["call_id"])

    form_file = (
        Path(FORMS_DIR)
        / f"{call_id}.txt"
    )

    if form_file.exists():

        original_text = form_file.read_text(
            encoding="utf-8"
        )

    else:

        original_text = "找不到原始會辦單"

    st.divider()

    st.subheader(
        f"📞 Call ID : {call_id}"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            <div style="
                background-color:#f5f5f5;
                padding:15px;
                border-radius:10px;
                border:1px solid #ddd;
            ">
            <h4>原會辦單</h4>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.text_area(
            "",
            original_text,
            height=220,
            key=f"ori_{call_id}"
        )

    with col2:

        st.markdown(
            """
            <div style="
                background-color:#eaf6ff;
                padding:15px;
                border-radius:10px;
                border:1px solid #ddd;
            ">
            <h4>LLM 擷取結果</h4>
            </div>
            """,
            unsafe_allow_html=True
        )

        extracted = f"""
問題描述：
{row["problem_description"]}

--------------------------------

需求內容：
{row["request_content"]}
"""

        st.text_area(
            "",
            extracted,
            height=220,
            key=f"pred_{call_id}"
        )
