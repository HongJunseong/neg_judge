import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import pandas as pd

from stage3_dataset import Stage3Dataset
from stage3_model import Stage3NegligenceClassifier

# ─── 경로 설정 ───────────────────────────────────────────
CHECKPOINT_DIR = "./checkpoints4"
PF_PATH       = os.path.join(CHECKPOINT_DIR, "train_first_preds_ep43.pth")
PF_VAL_PATH   = os.path.join(CHECKPOINT_DIR, "val_first_preds_ep43.pth")

CHECKPOINT_DIR_2 = "./checkpoints4_result"
SECOND_PATH      = os.path.join(CHECKPOINT_DIR_2, "train_stage2_preds_ep42.pth")
SECOND_VAL_PATH  = os.path.join(CHECKPOINT_DIR_2, "val_stage2_preds_ep42.pth")

LABEL_CSV_PATH = "./train_data_grouped_with_class.csv"

# ─── 하이퍼파라미터 ───────────────────────────────────────
BATCH_SIZE = 32
NUM_EPOCHS = 30
LR         = 1e-4
WD         = 1e-4
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

# ─── 데이터셋 & 로더 ─────────────────────────────────────
train_ds = Stage3Dataset(PF_PATH,      SECOND_PATH,     LABEL_CSV_PATH)
val_ds   = Stage3Dataset(PF_VAL_PATH,  SECOND_VAL_PATH, LABEL_CSV_PATH)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

# ─── 모델 / 옵티마이저 / 손실함수 ─────────────────────────
model     = Stage3NegligenceClassifier(num_classes=11).to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WD)
criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

# ─── 체크포인트 폴더 & CSV 파일 경로 ────────────────────────
os.makedirs(CHECKPOINT_DIR_2, exist_ok=True)
BEST_PATH    = os.path.join(CHECKPOINT_DIR_2, "stage3_best.pth")
METRICS_CSV  = os.path.join(CHECKPOINT_DIR_2, "stage3_metrics.csv")

# ─── 기록용 리스트 초기화 ─────────────────────────────────
history = {
    "epoch": [],
    "train_loss": [], "train_acc": [],
    "val_loss": [],   "val_acc": []
}

best_val_acc = 0.0

for epoch in range(1, NUM_EPOCHS+1):
    # ---- Train ----
    model.train()
    total, correct, train_loss = 0, 0, 0
    for pf, feat, a, b, label in tqdm(train_loader, desc=f"[Epoch {epoch}] Train"):
        pf, feat, a, b, label = (x.to(DEVICE) for x in (pf, feat, a, b, label))
        logits = model(pf, feat, a, b)
        loss = criterion(logits, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * pf.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == label).sum().item()
        total += label.size(0)

    train_acc  = correct / total
    train_loss = train_loss / total

    # ---- Validation ----
    model.eval()
    val_total, val_correct, val_loss = 0, 0, 0
    with torch.no_grad():
        for pf, feat, a, b, label in val_loader:
            pf, feat, a, b, label = (x.to(DEVICE) for x in (pf, feat, a, b, label))
            logits = model(pf, feat, a, b)
            loss = criterion(logits, label)

            val_loss   += loss.item() * pf.size(0)
            pred        = logits.argmax(dim=1)
            val_correct += (pred == label).sum().item()
            val_total   += label.size(0)

    val_acc  = val_correct / val_total
    val_loss = val_loss / val_total

    print(f"[Epoch {epoch}] "
          f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.3f} || "
          f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.3f}")

    # ---- Best 모델 저장 ----
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc
        }, BEST_PATH)
        print(f"👉 New best model saved (epoch {epoch}, val_acc {val_acc:.3f}) at\n   {BEST_PATH}")

    # ---- 메트릭 기록 ----
    history["epoch"].append(epoch)
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

# ─── CSV로 저장 ───────────────────────────────────────────
df = pd.DataFrame(history)
df.to_csv(METRICS_CSV, index=False)
print(f"✅ Saved training metrics to {METRICS_CSV}")
print(f"Training complete. Best Val Acc: {best_val_acc:.3f}")