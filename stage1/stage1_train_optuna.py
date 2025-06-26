# optuna_train.py (updated)

import os
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

from stage1_dataset import MultiFrameDataset
from stage1_model import MultiFrameClassifier
from stage1_utils import accuracy
from stage1_config import EPOCHS

# 체크포인트 폴더 생성
CHECKPOINT_DIR = "/checkpoints.."
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터로더 생성 함수 (학습/검증 구분)
def get_data_loader(image_root, label_txt, batch_size, is_train=True):
    if is_train:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    dataset = MultiFrameDataset(
        image_root=image_root,
        label_txt=label_txt,
        transform=transform
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=4
    )

# Optuna objective function with pruning
def objective(trial):
    # Initialize best validation accuracy for checkpointing
    best_val = 0.0

    # 하이퍼파라미터 탐색 범위
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "AdamW"])
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-4, 1e-3)
    dropout_rate = trial.suggest_uniform("dropout_rate", 0.0, 0.5)
    step_size = trial.suggest_int("step_size", 5, 20)
    gamma = trial.suggest_uniform("gamma", 0.1, 0.9)

    # 데이터로더
    train_loader = get_data_loader(
        image_root="data/train",
        label_txt="dataset_txt/train_accident_place.txt", # label 포함된 train dataset의 txt
        batch_size=batch_size,
        is_train=True
    )
    valid_loader = get_data_loader(
        image_root="data/val",
        label_txt="dataset_txt/val_accident_place.txt", # label 포함된 validation dataset의 txt
        batch_size=batch_size,
        is_train=False
    )

    # 모델, 손실, 옵티마이저, 스케줄러
    model = MultiFrameClassifier(dropout_rate=dropout_rate).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # epoch별 학습 및 평가
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_pbar = tqdm(train_loader, desc=f"Trial {trial.number} Epoch {epoch+1}/{EPOCHS} [Train]")
        for frames, labels in train_pbar:
            frames = [f.to(DEVICE) for f in frames]
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_pbar.set_postfix(loss=loss.item())

        # Scheduler step
        scheduler.step()

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_pbar = tqdm(valid_loader, desc=f"Trial {trial.number} Epoch {epoch+1}/{EPOCHS} [Val]")
        for frames, labels in val_pbar:
            frames = [f.to(DEVICE) for f in frames]
            labels = labels.to(DEVICE)
            outputs = model(frames)
            val_correct += accuracy(outputs, labels) * labels.size(0)
            val_total += labels.size(0)
            val_pbar.set_postfix(val_acc=f"{val_correct/val_total:.4f}")
        val_acc = val_correct / val_total

        # performance reporting & pruning
        trial.report(val_acc, epoch)
        if val_acc > best_val:
            best_val = val_acc
            # best checkpoint for this trial
            ckpt_path = os.path.join(
                CHECKPOINT_DIR,
                f"trial{trial.number}_best_epoch{epoch+1}_acc{best_val:.4f}.pth"
            )
            torch.save(model.state_dict(), ckpt_path)
        if trial.should_prune():
            print(f"Pruned Trial {trial.number} at epoch {epoch+1}, best_acc={best_val:.4f}")
            raise optuna.exceptions.TrialPruned()

    # pruning 없이 완료된 trial 최종 성능 반환
    return best_val

# CSV 저장용 콜백 정의
def save_trials_to_csv(study, trial):
    df = study.trials_dataframe()
    df.to_csv("optuna_trials_results.csv", index=False)
    print(f"[INFO] Saved {len(df)} trials to optuna_trials_results.csv")

def stop_after_n_complete(study, trial):
    complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if complete >= 30:
        study.stop()

if __name__ == "__main__":
    # Pruner 설정
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    study = optuna.create_study(study_name="my_study",
                                storage="sqlite:///optuna_study.db",
                                load_if_exists=True,
                                direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=30, timeout=7200,
                   callbacks=[save_trials_to_csv, stop_after_n_complete])

    # 결과 저장
    import pandas as pd
    df = study.trials_dataframe()
    df.to_csv("optuna_trials_results.csv", index=False)

    # 최적 결과 출력
    best = study.best_trial
    print(f"Best Trial #{best.number}: acc={best.value:.4f}, params={best.params}")

