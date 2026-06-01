import pandas as pd
import streamlit as st
from pathlib import Path

CSV_FILE = "results/extracted_calls.csv"
FORMS_DIR = "data/forms"

df = pd.read_csv(CSV_FILE)

st.set_page_config(
    page_title="會辦單比對",
    layout="wide"
)

st.title("會辦單擷取結果比對")

index = st.number_input(
    "選擇資料",
    min_value=1,
    max_value=len(df),
    value=1
)

row = df.iloc[index - 1]

call_id = str(row["call_id"])

st.header(f"Call ID：{call_id}")

form_path = Path(FORMS_DIR) / f"{call_id}.txt"

if form_path.exists():
    original_form = form_path.read_text(
        encoding="utf-8"
    )
else:
    original_form = "找不到原會辦單"

col1, col2 = st.columns(2)

with col1:

    st.subheader("原會辦單")

    st.text_area(
        "",
        original_form,
        height=600
    )

with col2:

    st.subheader("LLM 擷取結果")

    extracted_text = f"""
問題描述：
{row["problem_description"]}

需求內容：
{row["request_content"]}
"""

    st.text_area(
        "",
        extracted_text,
        height=600
    )
