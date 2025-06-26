# collect_stage2_metrics.py

import os
import torch
import pandas as pd
from torch.nn import CrossEntropyLoss

import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

# 설정: 필요에 따라 경로와 에포크 수를 조정하세요
CHECKPOINT_DIR = "./checkpoints4_result"
EPOCHS = 60

def compute_topk(logits, labels, k):
    topk = logits.topk(k, dim=1).indices
    return (topk == labels.unsqueeze(1)).any(dim=1).float().mean().item()

def main():
    criterion = CrossEntropyLoss(label_smoothing=0.1)

    records = {
        'epoch': [],
        'train_feat_top1': [], 'train_feat_top3': [],
        'train_a_top1': [],    'train_a_top3': [],
        'train_b_top1': [],    'train_b_top3': [],
        'val_feat_top1': [],   'val_feat_top3': [],
        'val_a_top1': [],      'val_a_top3': [],
        'val_b_top1': [],      'val_b_top3': [],
    }

    for ep in range(1, EPOCHS + 1):
        # ---- Train preds ----
        tr = torch.load(os.path.join(CHECKPOINT_DIR, f"train_stage2_preds_ep{ep}.pth"))
        feat_logits = tr['feat_logits']
        a_logits    = tr['a_logits']
        b_logits    = tr['b_logits']
        feat_labels = tr['feat_labels']
        a_labels    = tr['a_labels']
        b_labels    = tr['b_labels']

        records['epoch'].append(ep)
        records['train_feat_top1'].append((feat_logits.argmax(1) == feat_labels).float().mean().item())
        records['train_feat_top3'].append(compute_topk(feat_logits, feat_labels, 3))
        records['train_a_top1'].append((a_logits.argmax(1) == a_labels).float().mean().item())
        records['train_a_top3'].append(compute_topk(a_logits, a_labels, 3))
        records['train_b_top1'].append((b_logits.argmax(1) == b_labels).float().mean().item())
        records['train_b_top3'].append(compute_topk(b_logits, b_labels, 3))

        # ---- Val preds ----
        vr = torch.load(os.path.join(CHECKPOINT_DIR, f"val_stage2_preds_ep{ep}.pth"))
        feat_logits = vr['feat_logits']
        a_logits    = vr['a_logits']
        b_logits    = vr['b_logits']
        feat_labels = vr['feat_labels']
        a_labels    = vr['a_labels']
        b_labels    = vr['b_labels']

        records['val_feat_top1'].append((feat_logits.argmax(1) == feat_labels).float().mean().item())
        records['val_feat_top3'].append(compute_topk(feat_logits, feat_labels, 3))
        records['val_a_top1'].append((a_logits.argmax(1) == a_labels).float().mean().item())
        records['val_a_top3'].append(compute_topk(a_logits, a_labels, 3))
        records['val_b_top1'].append((b_logits.argmax(1) == b_labels).float().mean().item())
        records['val_b_top3'].append(compute_topk(b_logits, b_labels, 3))

        print(f"Epoch {ep} metrics collected.")

    df = pd.DataFrame(records)
    csv_out = os.path.join(CHECKPOINT_DIR, "stage2_metrics.csv")
    df.to_csv(csv_out, index=False)
    print(f"Stage2 metrics saved to {csv_out}")

if __name__ == "__main__":
    main()
