# test_stage2.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from stage2_dataset import Stage2Dataset
from stage2_model_slowfast50 import Stage2Model
from stage2_config import BATCH_SIZE, NUM_CLASSES, NUM_FEATURES, NUM_A_PROGRESS, NUM_B_PROGRESS

# 설정: 파일 경로 및 하이퍼파라미터
# Stage1 Test logits (.pth)에 저장된 keys, logits, labels

PF_LOGITS_PATH   = "./checkpoints.../test_first_preds.pth"

# Stage2 에서 학습된  best model
STAGE2_CKPT_PATH = "./checkpoints.../stage2_best.pth"

# Stage2 test 예측값 저장 경로
OUTPUT_PATH      = "./checkpoints.../test_stage2_preds.pth"

# Test 데이터셋 경로 및 레이블 매핑 파일
IMAGE_ROOT       = "data/test"
FEATURE_TXT      = "dataset_txt/test_accident_place_feature_mapped.txt"
A_TXT            = "dataset_txt/test_vehicle_a_progress_info_mapped.txt"
B_TXT            = "dataset_txt/test_vehicle_b_progress_info_mapped.txt"


# Stage1 Test logits 불러오기
pf_data = torch.load(PF_LOGITS_PATH, map_location="cpu")
pf_logits = pf_data["logits"]  # tensor [N, num_pf]


# Test 데이터 전처리 및 DataLoader
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

test_ds = Stage2Dataset(
    image_root=IMAGE_ROOT,
    feature_txt=FEATURE_TXT,
    a_txt=A_TXT,
    b_txt=B_TXT,
    preds_path=PF_LOGITS_PATH,
    transform=test_transform,
    random_sampling=False
)
test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# Stage2 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Stage2Model(
    num_pf=NUM_CLASSES,
    num_feat=NUM_FEATURES,
    num_a=NUM_A_PROGRESS,
    num_b=NUM_B_PROGRESS
).to(device)

checkpoint = torch.load(STAGE2_CKPT_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Inference 및 성능 평가
def compute_topk(logits, labels, k=3):
    topk = logits.topk(k, dim=1).indices
    return (topk == labels.unsqueeze(1)).any(dim=1).float().mean().item()

all_feat_logits, all_a_logits, all_b_logits = [], [], []
all_feat_labels, all_a_labels, all_b_labels = [], [], []
all_ids = []

corr_f1 = corr_f3 = 0
corr_a1 = corr_a3 = 0
corr_b1 = corr_b3 = 0
total   = 0

with torch.no_grad():
    for frames, pf_in, f_lbl, a_lbl, b_lbl, ids in test_loader:
        # PF logits slicing
        batch_size = f_lbl.size(0)
        start = total
        end   = total + batch_size
        pf_batch = pf_logits[start:end].to(device)
        total += batch_size

        # 영상 입력
        video = torch.stack(frames, dim=2).to(device)
        out_f, out_a, out_b = model([video[:,:,::4], video], pf_batch)

        # accumulate outputs
        all_feat_logits.append(out_f.cpu())
        all_a_logits.append(out_a.cpu())
        all_b_logits.append(out_b.cpu())
        all_feat_labels.append(f_lbl)
        all_a_labels.append(a_lbl)
        all_b_labels.append(b_lbl)
        all_ids.extend(ids)

        # Top-1
        corr_f1 += (out_f.argmax(1) == f_lbl.to(device)).sum().item()
        corr_a1 += (out_a.argmax(1) == a_lbl.to(device)).sum().item()
        corr_b1 += (out_b.argmax(1) == b_lbl.to(device)).sum().item()
        # Top-3
        corr_f3 += compute_topk(out_f, f_lbl.to(device), 3) * batch_size
        corr_a3 += compute_topk(out_a, a_lbl.to(device), 3) * batch_size
        corr_b3 += compute_topk(out_b, b_lbl.to(device), 3) * batch_size

# 최종 Accuracy
feat_t1 = corr_f1 / total; feat_t3 = corr_f3 / total
a_t1    = corr_a1 / total; a_t3    = corr_a3 / total
b_t1    = corr_b1 / total; b_t3    = corr_b3 / total
print(f"Test Feature   ▶ Top-1: {feat_t1:.4f}, Top-3: {feat_t3:.4f}")
print(f"Test A Direction▶ Top-1: {a_t1:.4f}, Top-3: {a_t3:.4f}")
print(f"Test B Direction▶ Top-1: {b_t1:.4f}, Top-3: {b_t3:.4f}")

# 결과 저장
output = {
    "ids":            all_ids,
    "feat_logits":    torch.cat(all_feat_logits, dim=0),
    "a_logits":       torch.cat(all_a_logits,    dim=0),
    "b_logits":       torch.cat(all_b_logits,    dim=0),
    "feat_labels":    torch.cat(all_feat_labels, dim=0),
    "a_labels":       torch.cat(all_a_labels,    dim=0),
    "b_labels":       torch.cat(all_b_labels,    dim=0),
}
torch.save(output, OUTPUT_PATH)
print(f"[Saved TEST preds] {OUTPUT_PATH}")
