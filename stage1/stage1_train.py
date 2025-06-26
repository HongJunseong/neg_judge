import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from stage1.stage1_dataset import MultiFrameDataset
from stage1.stage1_model import MultiFrameClassifier
from stage1_utils import accuracy
from stage1.stage1_config import BATCH_SIZE, EPOCHS, LR, FRAMES_PER_SAMPLE, NUM_CLASSES
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

# 체크포인트 디렉토리
CHECKPOINT_DIR = "./checkpoints.."
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.RandomRotation(5),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    # dataset & loader
    train_ds = MultiFrameDataset("data/train", "dataset_txt/train_accident_place.txt",
                                 train_transform, FRAMES_PER_SAMPLE)
    val_ds   = MultiFrameDataset("data/val", "dataset_txt/val_accident_place.txt",
                                 val_transform, FRAMES_PER_SAMPLE)
    
    print(">>> TRAIN 샘플 수:", len(train_ds))
    print(">>>  VAL 샘플 수:", len(val_ds))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # model / optimizer / scheduler / loss
    model = MultiFrameClassifier(NUM_CLASSES, FRAMES_PER_SAMPLE).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=19, gamma=0.37)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # resume
    best_acc = 0.0
    start_ep = 0
    last_ckpt = os.path.join(CHECKPOINT_DIR, "last_checkpoint.pth")
    if os.path.exists(last_ckpt):
        ck = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ck['model_state_dict'])
        optimizer.load_state_dict(ck['optimizer_state_dict'])
        start_ep = ck['epoch'] + 1
        best_acc = ck.get('best_acc', 0.0)
        print(f"[RESUME] Epoch {start_ep}, Best Acc {best_acc:.4f}")

    for epoch in range(start_ep, EPOCHS):
        # ---- train ----
        model.train()
        running_loss = 0.0
        all_logits_train, all_labels_train, all_keys_train = [], [], []

        for frames, labels, keys in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} Train"):
            frames = [f.to(device) for f in frames]
            labels = labels.to(device)
            preds = model(frames)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 예측 결과 수집 (detach() 해서 저장)
            all_logits_train.append(preds.detach().cpu())
            all_labels_train.append(labels.cpu())
            all_keys_train.extend(keys)

        print(f"Train Loss: {running_loss / len(train_loader):.4f}")

        # ---- save train preds ----
        train_save = {
            'keys': all_keys_train,
            'logits': torch.cat(all_logits_train, dim=0),
            'labels': torch.cat(all_labels_train, dim=0)
        }
        train_pred_path = os.path.join(CHECKPOINT_DIR, f"train_first_preds_ep{epoch+1}.pth")
        torch.save(train_save, train_pred_path)
        print(f"[Saved TRAIN preds] {train_pred_path}")

        # ---- validation ----
        model.eval()
        total, correct = 0, 0
        all_logits_val, all_labels_val, all_keys_val = [], [], []

        with torch.no_grad():
            for frames, labels, keys in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} Validation"):
                frames = [f.to(device) for f in frames]
                labels = labels.to(device)
                preds = model(frames)

                acc = accuracy(preds, labels)
                correct += acc * labels.size(0)
                total   += labels.size(0)

                all_logits_val.append(preds.cpu())
                all_labels_val.append(labels.cpu())
                all_keys_val.extend(keys)

        val_acc = correct / total
        print(f"Val Accuracy: {val_acc:.4f}")

        # ---- save val preds ----
        val_save = {
            'keys': all_keys_val,
            'logits': torch.cat(all_logits_val, dim=0),
            'labels': torch.cat(all_labels_val, dim=0)
        }
        val_pred_path = os.path.join(CHECKPOINT_DIR, f"val_first_preds_ep{epoch+1}.pth")
        torch.save(val_save, val_pred_path)
        print(f"[Saved VAL preds] {val_pred_path}")

        # ---- checkpoints ----
        ck = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc
        }
        torch.save(ck, last_ckpt)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))
            print(f"[🏆 New best model] Acc {best_acc:.4f}")

        scheduler.step()

if __name__ == "__main__":
    main()
