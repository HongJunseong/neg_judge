import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import SGD
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

from stage3_dataset import Stage3Dataset
from stage3_model import Stage3NegligenceClassifier

# --- 경로 설정 ---
CHECKPOINT_DIR   = "./checkpoints4"
PF_PATH         = os.path.join(CHECKPOINT_DIR,   "train_first_preds_ep43.pth")
PF_VAL_PATH     = os.path.join(CHECKPOINT_DIR,   "val_first_preds_ep43.pth")

CHECKPOINT_DIR = "./checkpoints4_result"
SECOND_PATH = os.path.join(CHECKPOINT_DIR, "train_stage2_preds_ep42.pth")
SECOND_VAL_PATH = os.path.join(CHECKPOINT_DIR, "val_stage2_preds_ep42.pth")

LABEL_CSV_PATH = "./train_data_grouped_with_class.csv"

# --- 하이퍼파라미터 ---
BATCH_SIZE  = 32
NUM_EPOCHS  = 50
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
WD          = 1e-4

# --- 클래스 이름 (11개: 0:100, 10:90, …, 100:0) ---
class_names = [f"{i*10}:{100-i*10}" for i in range(11)]
num_classes = len(class_names)

# --- 데이터셋 & 로더 ---
train_ds = Stage3Dataset(PF_PATH,       SECOND_PATH,     LABEL_CSV_PATH)
val_ds   = Stage3Dataset(PF_VAL_PATH,   SECOND_VAL_PATH, LABEL_CSV_PATH)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

# --- 모델 & 옵티마이저 & 손실함수 ---
model     = Stage3NegligenceClassifier(num_classes=num_classes).to(DEVICE)
optimizer = SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=WD)
criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

for epoch in range(1, NUM_EPOCHS+1):
    # --- Train ---
    model.train()
    for pf, feat, a, b, label in tqdm(train_loader, desc=f"[Epoch {epoch}] Train"):
        pf, feat, a, b, label = (x.to(DEVICE) for x in (pf, feat, a, b, label))
        logits = model(pf, feat, a, b)
        loss = criterion(logits, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # --- Validation & Prediction 수집 ---
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for pf, feat, a, b, label in tqdm(val_loader, desc=f"[Epoch {epoch}] Val"):
            pf, feat, a, b, label = (x.to(DEVICE) for x in (pf, feat, a, b, label))
            logits = model(pf, feat, a, b)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(label.cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    # --- 1) Confusion Matrix & Classification Report ---
    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(
        labels, preds,
        labels=list(range(num_classes)),
        target_names=class_names,
        digits=3
    ))

    # --- 2) One-off Accuracy (±1 bin) ---
    one_off = np.mean(np.abs(preds - labels) <= 1)
    print(f"One-off Accuracy (±1 class): {one_off:.3f}")

    # --- 3) 근접도 지표: Error Distribution & MAE/RMSE ---
    errors = np.abs(preds - labels)  # 클래스 인덱스 차이
    mae  = errors.mean()
    rmse = np.sqrt((errors**2).mean())
    print(f"\nMean Absolute Error (in class bins): {mae:.3f}")
    print(f"Root Mean Squared Error (in class bins): {rmse:.3f}")

    # 오차 분포 (거리별 비율)
    print("\nError distance distribution:")
    total = len(errors)
    for d in range(errors.max()+1):
        cnt = np.sum(errors == d)
        print(f"  ±{d} bins: {cnt} samples ({cnt/total:.3f})")

    # 클래스별 평균 오차
    print("\nTrue-class별 평균 오차:")
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            print(f"  {class_names[c]}: {errors[mask].mean():.3f}")
    print("="*60)
