import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from dataset import MultiFrameDataset
from stage1.stage1_model import MultiFrameClassifier
from stage1_utils import accuracy  # 기존에 쓰시던 Top-1 계산 함수
from config import BATCH_SIZE, FRAMES_PER_SAMPLE, NUM_CLASSES

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 저장된 best model 경로
    checkpoint_path = "./checkpoints4/best_model.pth"
    assert os.path.exists(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"

    # val/test 전처리와 동일하게 맞춰주세요
    normalize = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    test_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize,
    ])

    # Test dataset 경로와 리스트 파일을 알맞게 설정
    test_ds = MultiFrameDataset(
        "data2/test",                   # test 프레임 폴더
        "tsn_dataset/test_accident_place.txt",  # test 리스트
        test_transform,
        FRAMES_PER_SAMPLE
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 모델 초기화 및 weight 로드
    model = MultiFrameClassifier(NUM_CLASSES, FRAMES_PER_SAMPLE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()

    total = 0
    correct_top1 = 0
    correct_top3 = 0

    all_logits = []
    all_labels = []
    all_keys = []

    with torch.no_grad():
        for frames, labels, keys in test_loader:
            # frames: list of T tensors [B,3,224,224]
            frames = [f.to(device) for f in frames]
            labels = labels.to(device)

            logits = model(frames)        # [B, C]
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_keys.extend(keys)

            # Top-1
            preds1 = logits.argmax(dim=1)
            correct_top1 += (preds1 == labels).sum().item()
            # Top-3
            top3 = logits.topk(3, dim=1).indices   # [B,3]
            correct_top3 += (top3 == labels.unsqueeze(1)).any(dim=1).sum().item()
            total += labels.size(0)

    # 전체 Accuracy 계산
    top1_acc = correct_top1 / total
    top3_acc = correct_top3 / total
    print(f"Test Top-1 Accuracy: {top1_acc:.4f}")
    print(f"Test Top-3 Accuracy: {top3_acc:.4f}")

    # logits, labels, keys 저장 (필요시)
    out = {
        "keys": all_keys,
        "logits": torch.cat(all_logits, dim=0),
        "labels": torch.cat(all_labels, dim=0)
    }
    save_path = os.path.join(os.path.dirname(checkpoint_path), "test_first_preds.pth")
    torch.save(out, save_path)
    print(f"[Saved TEST preds] {save_path}")

if __name__ == "__main__":
    main()