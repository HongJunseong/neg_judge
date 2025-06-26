# test_stage3.py

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

from stage3_dataset import Stage3Dataset
from stage3_model import Stage3NegligenceClassifier

# ─── 경로 설정 ───────────────────────────────────────────────
# Stage1 Test preds (ids, logits, labels)
PF_TEST_PATH     = "./checkpoints.../model.pth"
# Stage2 Test preds (ids, feat_logits, a_logits, b_logits, labels)

SECOND_TEST_PATH = "./checkpoints...../model.pth"

# Stage3 모델 체크포인트 (model_state_dict 저장된 파일)
STAGE3_CKPT_PATH = "./checkpoints....../model.pth"  # 실제 경로로 바꾸기

# 샘플 → 클래스 매핑 CSV (Stage3Dataset 내부에서 라벨을 로드할 때 사용)
LABEL_CSV_PATH   = "./train_data_grouped_with_class.csv"

# ─── 하이퍼파라미터 ─────────────────────────────────────────
BATCH_SIZE = 32
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

# ─── 클래스 이름 정의 ────────────────────────────────────────
# 0:100, 10:90, …, 100:0
class_names = [f"{i*10}:{100-i*10}" for i in range(11)]
num_classes = len(class_names)

# ─── Dataset & DataLoader ────────────────────────────────────
test_ds = Stage3Dataset(PF_TEST_PATH, SECOND_TEST_PATH, LABEL_CSV_PATH)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# ─── 모델 로드 ───────────────────────────────────────────────
model = Stage3NegligenceClassifier(num_classes=num_classes).to(DEVICE)
ckpt  = torch.load(STAGE3_CKPT_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# ─── Inference & 결과 수집 ───────────────────────────────────
all_logits = []
all_preds  = []
all_labels = []

with torch.no_grad():
    for pf, feat, a, b, label in test_loader:
        pf, feat, a, b = [x.to(DEVICE) for x in (pf, feat, a, b)]
        logits = model(pf, feat, a, b)              # [B, 11]
        preds  = logits.argmax(dim=1)              # [B]

        all_logits.append(logits.cpu().numpy())
        all_preds .append(preds.cpu().numpy())
        all_labels.append(label.numpy())

# 배열 형태로 합치기
logits = np.concatenate(all_logits, axis=0)  # [N, 11]
preds  = np.concatenate(all_preds,  axis=0)  # [N]
labels = np.concatenate(all_labels,axis=0)   # [N]

# ─── Confusion Matrix & Classification Report ───────────
cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
print("Confusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(
    labels, preds,
    labels=list(range(num_classes)),
    target_names=class_names,
    digits=3
))

# ─── One-off Accuracy (±1 bin) ───────────────────────────
one_off = np.mean(np.abs(preds - labels) <= 1)
print(f"\nOne-off Accuracy (±1 class): {one_off:.3f}")

# ─── Error Distribution & MAE/RMSE ────────────────────────
errors = np.abs(preds - labels)
mae   = errors.mean()
rmse  = np.sqrt((errors**2).mean())
print(f"\nMean Absolute Error (in class bins): {mae:.3f}")
print(f"Root Mean Squared Error (in class bins): {rmse:.3f}")

print("\nError distance distribution:")
total = len(errors)
for d in range(errors.max()+1):
    cnt = int((errors == d).sum())
    print(f"  ±{d} bins: {cnt} samples ({cnt/total:.3f})")

print("\nMean error per true class:")
for i, name in enumerate(class_names):
    mask = (labels == i)
    if mask.sum() > 0:
        print(f"  {name:7s}: {errors[mask].mean():.3f}")
