from pathlib import Path
from collections import Counter
import json
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


EMBEDDINGS_PATH = Path("processed/embeddings/call_embeddings.npz")
OUTPUT_DIR = Path("processed/evaluation")

REPORT_PATH = OUTPUT_DIR / "loocv_report.json"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix.csv"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = np.load(EMBEDDINGS_PATH, allow_pickle=True)

    X = data["vectors"]
    y_text = data["labels"]
    call_ids = data["call_ids"]

    label_counter = Counter(y_text)

    # 方法：Leave-One-Out Cross Validation
    # 每次留一筆當測試，其餘當訓練。
    #
    # 重要限制：
    # 如果某類別只有 1 筆，拿它當 test 時，
    # train 裡就完全沒有這個類別，模型不可能預測出該類。
    # 因此這裡會跳過 singleton 類別的 LOOCV。
    valid_indices = [
        i for i, label in enumerate(y_text)
        if label_counter[label] >= 2
    ]

    skipped_indices = [
        i for i, label in enumerate(y_text)
        if label_counter[label] < 2
    ]

    y_true_all = []
    y_pred_all = []
    details = []

    for test_idx in valid_indices:
        train_indices = [
            i for i in range(len(X))
            if i != test_idx
        ]

        X_train = X[train_indices]
        X_test = X[[test_idx]]

        y_train_text = y_text[train_indices]
        y_test_text = y_text[test_idx]

        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train_text)

        clf = SVC(
            kernel="linear",
            probability=True,
            class_weight="balanced"
        )

        clf.fit(X_train, y_train)

        pred_encoded = clf.predict(X_test)[0]
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]

        y_true_all.append(y_test_text)
        y_pred_all.append(pred_label)

        details.append(
            {
                "call_id": str(call_ids[test_idx]),
                "true_label": str(y_test_text),
                "pred_label": str(pred_label),
                "correct": bool(y_test_text == pred_label)
            }
        )

    accuracy = accuracy_score(y_true_all, y_pred_all) if y_true_all else 0.0

    labels_sorted = sorted(set(y_true_all) | set(y_pred_all))

    report = {
        "method": "Leave-One-Out Cross Validation",
        "note": "類別只有一筆資料時，該筆 LOOCV 會被跳過，因為訓練集中沒有該類別。",
        "summary": {
            "total_samples": int(len(X)),
            "evaluated_samples": int(len(valid_indices)),
            "skipped_singleton_samples": int(len(skipped_indices)),
            "accuracy": float(accuracy)
        },
        "skipped_singleton_cases": [
            {
                "call_id": str(call_ids[i]),
                "label_name": str(y_text[i])
            }
            for i in skipped_indices
        ],
        "details": details,
        "classification_report": classification_report(
            y_true_all,
            y_pred_all,
            labels=labels_sorted,
            zero_division=0,
            output_dict=True
        ) if y_true_all else {}
    }

    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if y_true_all:
        cm = confusion_matrix(
            y_true_all,
            y_pred_all,
            labels=labels_sorted
        )

        with CONFUSION_MATRIX_PATH.open("w", encoding="utf-8") as f:
            f.write("," + ",".join(labels_sorted) + "\n")

            for label, row in zip(labels_sorted, cm):
                f.write(label + "," + ",".join(map(str, row)) + "\n")

    print("LOOCV 評估完成")
    print(f"Report：{REPORT_PATH}")
    print(f"Confusion Matrix：{CONFUSION_MATRIX_PATH}")
    print()
    print("摘要：")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
