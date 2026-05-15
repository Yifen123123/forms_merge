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

    # ---------- step1 ----------
    # remove repeated fillers

    for pattern in REPEAT_PATTERNS:
        text = re.sub(pattern, "", text)

    for word in FILLER_WORDS:
        text = text.replace(word, "")

    # ---------- step2 ----------
    # line-by-line cleaning

    cleaned_lines = []

    lines = text.split("\n")

    for line in lines:

        line = line.strip()

        if not line:
            continue

        # speaker only
        #
        # L:
        # R:
        #
        if re.fullmatch(r"[LR]\s*:", line):
            continue

        # speaker + punctuation only
        #
        # L:，
        # R:。
        # L:...
        #
        if re.fullmatch(
            r"[LR]\s*:\s*[，。！？,.!?、…\s]*",
            line
        ):
            continue

        # punctuation only
        #
        # ...
        # ，，，，
        #
        if re.fullmatch(
            r"[，。！？,.!?、…\s]+",
            line
        ):
            continue

        cleaned_lines.append(line)

    # ---------- step3 ----------

    text = "\n".join(cleaned_lines)

    # collapse empty lines

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

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
