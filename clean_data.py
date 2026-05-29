from pathlib import Path
import re


INPUT_DIR = Path("data")
OUTPUT_DIR = Path("cleaned_data")


# 00:00:01.645 00:00:06.324
TIMESTAMP_PATTERN = re.compile(
    r"\d{2}:\d{2}:\d{2}\.\d{3}\s+\d{2}:\d{2}:\d{2}\.\d{3}"
)

DATE_PATTERN = re.compile(r"日期[:：]\s*\d{4}/\d{1,2}/\d{1,2}")
ATTENDEE_PATTERN = re.compile(r"出席者[:：]\s*左音軌\s*[,，]\s*右音軌")
MEETING_RECORD_PATTERN = re.compile(r"會議紀錄[:：]")

LEFT_SPEAKER_PATTERN = re.compile(r"^左音軌\s*[:：]\s*$")
RIGHT_SPEAKER_PATTERN = re.compile(r"^右音軌\s*[:：]\s*$")


def should_remove_header_line(line: str, line_index: int) -> bool:
    """
    移除固定格式的前幾行：
    1. 第一行：xxx.wav
    3. 第三行：日期：YYYY/MM/DD
    5. 第五行：出席者：左音軌, 右音軌
    7. 第七行：會議紀錄：
    """
    text = line.strip()

    # 第 1 行，通常是 xxx.wav
    if line_index == 0 and text.lower().endswith(".wav"):
        return True

    # 第 3 行
    if line_index == 2 and DATE_PATTERN.search(text):
        return True

    # 第 5 行
    if line_index == 4 and ATTENDEE_PATTERN.search(text):
        return True

    # 第 7 行
    if line_index == 6 and MEETING_RECORD_PATTERN.search(text):
        return True

    return False


def clean_file(input_path: Path, output_path: Path) -> None:
    with input_path.open("r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    # Step 1：先移除指定 header 行
    lines = []
    for idx, line in enumerate(raw_lines):
        if should_remove_header_line(line, idx):
            continue
        lines.append(line.strip())

    # Step 2：移除時間戳、空白行
    cleaned_lines = []
    for line in lines:
        line = TIMESTAMP_PATTERN.sub("", line).strip()

        if not line:
            continue

        cleaned_lines.append(line)

    # Step 3：處理「左音軌:」下一行合併成 L: 文字
    #       處理「右音軌:」下一行合併成 R: 文字
    final_lines = []
    i = 0

    while i < len(cleaned_lines):
        line = cleaned_lines[i].strip()

        if LEFT_SPEAKER_PATTERN.match(line):
            if i + 1 < len(cleaned_lines):
                text = cleaned_lines[i + 1].strip()
                final_lines.append(f"L: {text}")
                i += 2
            else:
                final_lines.append("L:")
                i += 1

        elif RIGHT_SPEAKER_PATTERN.match(line):
            if i + 1 < len(cleaned_lines):
                text = cleaned_lines[i + 1].strip()
                final_lines.append(f"R: {text}")
                i += 2
            else:
                final_lines.append("R:")
                i += 1

        else:
            # 如果原本已經是「左音軌: 文字」這種格式，也順便轉換
            line = re.sub(r"^左音軌\s*[:：]\s*", "L: ", line)
            line = re.sub(r"^右音軌\s*[:：]\s*", "R: ", line)
            final_lines.append(line)
            i += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(final_lines))


def clean_all_files() -> None:
    txt_files = list(INPUT_DIR.rglob("*.txt"))

    if not txt_files:
        print(f"找不到 txt 檔案，請確認資料夾是否存在：{INPUT_DIR}")
        return

    for input_path in txt_files:
        relative_path = input_path.relative_to(INPUT_DIR)
        output_path = OUTPUT_DIR / relative_path

        clean_file(input_path, output_path)

    print(f"清洗完成，共處理 {len(txt_files)} 份 txt 檔案")
    print(f"輸出位置：{OUTPUT_DIR}")


if __name__ == "__main__":
    clean_all_files()
