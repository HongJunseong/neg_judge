import os
import torch
import pandas as pd
from torch.nn import CrossEntropyLoss
from config import EPOCHS  # CHECKPOINT_DIR = "./checkpoints4"
from utils import accuracy  # Top-1 accuracy 계산 함수

CHECKPOINT_DIR = "./checkpoints4"

def main():
    criterion = CrossEntropyLoss(label_smoothing=0.1)

    epochs = []
    train_losses = []
    val_losses   = []
    train_accs   = []
    val_accs     = []

    for ep in range(1, EPOCHS+1):
        # Train preds 불러오기
        train_path = os.path.join(CHECKPOINT_DIR, f"train_first_preds_ep{ep}.pth")
        tr = torch.load(train_path)
        logits = tr['logits']
        labels = tr['labels']
        with torch.no_grad():
            loss = criterion(logits, labels).item()
            acc  = (logits.argmax(dim=1) == labels).float().mean().item()

        train_losses.append(loss)
        train_accs  .append(acc)

        # Val preds 불러오기
        val_path = os.path.join(CHECKPOINT_DIR, f"val_first_preds_ep{ep}.pth")
        vr = torch.load(val_path)
        logits = vr['logits']
        labels = vr['labels']
        with torch.no_grad():
            loss = criterion(logits, labels).item()
            acc  = (logits.argmax(dim=1) == labels).float().mean().item()

        val_losses.append(loss)
        val_accs  .append(acc)

        epochs.append(ep)
        print(f"Epoch {ep}: Train Loss={train_losses[-1]:.4f}, Acc={train_accs[-1]:.4f} | "
              f"Val Loss={val_losses[-1]:.4f}, Acc={val_accs[-1]:.4f}")

    # DataFrame 생성 및 CSV 저장
    df = pd.DataFrame({
        'epoch': epochs,
        'train_loss': train_losses,
        'val_loss':   val_losses,
        'train_acc':  train_accs,
        'val_acc':    val_accs,
    })
    csv_path = os.path.join(CHECKPOINT_DIR, "stage1_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved metrics to {csv_path}")

if __name__ == "__main__":
    main()