from pathlib import Path
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


EMBEDDINGS_PATH = Path("processed/embeddings/call_embeddings.npz")
MODEL_DIR = Path("models")

SVM_MODEL_PATH = MODEL_DIR / "svm_classifier.joblib"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.joblib"


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    data = np.load(EMBEDDINGS_PATH, allow_pickle=True)

    X = data["vectors"]
    y_text = data["labels"]

    # 方法：Label Encoding
    # 將文字 label 轉成 sklearn 可以訓練的數字 label。
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)

    # 方法：Linear SVM
    # small data + embedding 特徵時，linear SVM 通常比複雜模型穩定。
    clf = SVC(
        kernel="linear",
        probability=True,
        class_weight="balanced"
    )

    clf.fit(X, y)

    joblib.dump(clf, SVM_MODEL_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)

    print("SVM 訓練完成")
    print(f"模型：{SVM_MODEL_PATH}")
    print(f"Label encoder：{LABEL_ENCODER_PATH}")
    print(f"資料筆數：{len(X)}")
    print(f"類別數：{len(label_encoder.classes_)}")


if __name__ == "__main__":
    main()
