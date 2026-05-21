import json
import csv

# =========================
# 輸入與輸出檔案設定
# =========================
INPUT_JSONL = "input.jsonl"

# 第1個輸出：全部 true / pred
OUTPUT_ALL_CSV = "all_true_pred.csv"

# 第2個輸出：true != pred
OUTPUT_DIFF_CSV = "diff_true_pred.csv"


# =========================
# 讀取 jsonl
# =========================
all_rows = []
diff_rows = []

with open(INPUT_JSONL, "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, start=1):

        line = line.strip()

        # 跳過空行
        if not line:
            continue

        try:
            data = json.loads(line)

            true_value = data.get("true", "")
            pred_value = data.get("pred", "")

            row = {
                "true": true_value,
                "pred": pred_value
            }

            # 全部資料
            all_rows.append(row)

            # true != pred
            if str(true_value) != str(pred_value):
                diff_rows.append(row)

        except json.JSONDecodeError:
            print(f"[JSON格式錯誤] 第 {line_num} 行")
            continue


# =========================
# 輸出全部資料 CSV
# =========================
with open(OUTPUT_ALL_CSV, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=["true", "pred"])

    writer.writeheader()
    writer.writerows(all_rows)

print(f"已輸出全部資料 -> {OUTPUT_ALL_CSV}")


# =========================
# 輸出不同資料 CSV
# =========================
with open(OUTPUT_DIFF_CSV, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=["true", "pred"])

    writer.writeheader()
    writer.writerows(diff_rows)

print(f"已輸出 true != pred 的資料 -> {OUTPUT_DIFF_CSV}")


# =========================
# 統計資訊
# =========================
print("\n========== 統計 ==========")
print(f"總資料數: {len(all_rows)}")
print(f"不同資料數: {len(diff_rows)}")
print(f"相同率: {(1 - len(diff_rows)/len(all_rows))*100:.2f}%")
