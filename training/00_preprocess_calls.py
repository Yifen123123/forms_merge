from pathlib import Path
import re


INPUT_DIR = Path("calls")
OUTPUT_DIR = Path("training/processed_calls")

FILLER_WORDS = [
    "嗯", "呃", "啊", "欸", "誒", "喔", "哦", "噢",
    "嘛", "啦", "耶", "齁", "蛤", "痾", "ㄜ",
    "就是", "那個", "這個", "然後", "其實",
]

REPEAT_PATTERNS = [
    r"嗯+",
    r"呃+",
    r"啊+",
    r"欸+",
    r"喔+",
    r"哦+",
]


def clean_text(text: str) -> str:
    for pattern in REPEAT_PATTERNS:
        text = re.sub(pattern, "", text)

    for word in FILLER_WORDS:
        text = text.replace(word, "")

    # 清掉多餘空白
    text = re.sub(r"[ \t]+", " ", text)

    # 清掉過多換行
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(INPUT_DIR.glob("*.txt"))

    if not txt_files:
        print(f"No .txt files found in {INPUT_DIR}")
        return

    for file_path in txt_files:
        raw_text = file_path.read_text(encoding="utf-8")
        cleaned_text = clean_text(raw_text)

        output_path = OUTPUT_DIR / file_path.name
        output_path.write_text(cleaned_text, encoding="utf-8")

        print(f"Processed: {file_path} -> {output_path}")

    print(f"\nDone. Cleaned files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
