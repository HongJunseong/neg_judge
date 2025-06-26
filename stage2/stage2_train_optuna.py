# train_optuna.py

import os
import torch
import optuna
from torch.utils.data import DataLoader
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torchvision import transforms
from torchvision.transforms import RandAugment, RandomErasing
from tqdm import tqdm

from stage2_dataset import Stage2Dataset  # __getitem__ returns (frames, pf_logits, f_lbl, a_lbl, b_lbl, sample_id)
from stage2_config import *               # NUM_CLASSES, NUM_FEATURES, NUM_A_PROGRESS, NUM_B_PROGRESS, EPOCHS
from stage2_utils import FocalLoss

import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

# 1) 1차 모델 preds용 디렉토리 (for preds_path in Stage2Dataset)
FIRST_CHECKPOINT_DIR = "./checkpoints4"
# 2) OPTUNA 전용 체크포인트 디렉토리
OPTUNA_BASE_DIR = "./checkpoints_stage2_optuna_slowfast"
os.makedirs(OPTUNA_BASE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

N_TRIALS = 30

def save_preds(epoch, split, ids, feat_list, a_list, b_list, out_dir):
    """Train/Val epoch별 logits 파일 저장 (feat_list 등이 비어있으면 저장하지 않음)"""
    if not feat_list or not a_list or not b_list or not ids:
        return
    filepath = os.path.join(out_dir, f"{split}_stage2_preds_ep{epoch}.pth")
    torch.save({
        'ids': ids,
        'feat_logits': torch.cat(feat_list, dim=0),
        'a_logits':    torch.cat(a_list,    dim=0),
        'b_logits':    torch.cat(b_list,    dim=0),
    }, filepath)
    print(f"[Saved {split.upper()} preds @ epoch {epoch}] {filepath}")

def save_trials_to_csv(study, trial):
    df = study.trials_dataframe()
    df.to_csv("optuna_stage2_results_slowfast.csv", index=False)
    print(f"[INFO] Saved {len(df)} trials to optuna_stage2_results_slowfast.csv")

def stop_after_n_complete(study, trial):
    complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if complete >= N_TRIALS:
        study.stop()

def run_one_trial(hparams, trial):
    trial_idx = trial.number + 1
    # 디렉토리도 1-based 번호로 생성
    trial_dir = os.path.join(OPTUNA_BASE_DIR, f"trial_{trial_idx}")
    os.makedirs(trial_dir, exist_ok=True)

    preds_dir = os.path.join(trial_dir, "preds")
    os.makedirs(preds_dir, exist_ok=True)

    ckpt_path = os.path.join(trial_dir, "stage2_last.pth")

    # transforms 정의
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8,1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        RandAugment(num_ops=2, magnitude=9),
        transforms.ColorJitter(0.1,0.1,0.1,0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        RandomErasing(p=0.1, scale=(0.02,0.1)),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # 데이터셋 & 로더
    train_ds = Stage2Dataset(
        image_root='data/train',
        feature_txt='dataset_txt/train_accident_place_feature_mapped.txt',
        a_txt='dataset_txt/train_vehicle_a_progress_info_mapped.txt',
        b_txt='dataset_txt/train_vehicle_b_progress_info_mapped.txt',
        preds_path=os.path.join(FIRST_CHECKPOINT_DIR, 'train_first_preds_ep43.pth'),
        transform=train_transform,
        random_sampling=True
    )
    val_ds = Stage2Dataset(
        image_root='data/val',
        feature_txt='dataset_txt/val_accident_place_feature_mapped.txt',
        a_txt='dataset_txt/val_vehicle_a_progress_info_mapped.txt',
        b_txt='dataset_txt/val_vehicle_b_progress_info_mapped.txt',
        preds_path=os.path.join(FIRST_CHECKPOINT_DIR, 'val_first_preds_ep43.pth'),
        transform=val_transform,
        random_sampling=False
    )
    train_loader = DataLoader(train_ds, batch_size=hparams["batch_size"], shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=hparams["batch_size"], shuffle=False, num_workers=2)

    # 모델, 손실
    from stage2.stage2_model_slowfast50 import Stage2Model
    
    model = Stage2Model(
        num_pf=NUM_CLASSES,
        num_feat=NUM_FEATURES,
        num_a=NUM_A_PROGRESS,
        num_b=NUM_B_PROGRESS,
        dropout=hparams["dropout"]
    ).to(DEVICE)

    criterion = FocalLoss(alpha=hparams["focal_alpha"], gamma=hparams["focal_gamma"])

    # 옵티마이저
    if hparams["optimizer_name"] == "SGD":
        optimizer = SGD(
            model.parameters(),
            lr=hparams["lr"],
            momentum=0.9,
            weight_decay=hparams["weight_decay"]
        )
    else:
        optimizer = AdamW(
            model.parameters(),
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"]
        )

    # 스케줄러
    if hparams["scheduler_name"] == "StepLR":
        scheduler = StepLR(optimizer, step_size=hparams["step_size"], gamma=hparams["gamma"])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Resume 로직
    start_epoch = 1
    best_score  = -float("inf")
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ck['model_state_dict'])
        optimizer.load_state_dict(ck['optimizer_state_dict'])
        scheduler.load_state_dict(ck['scheduler_state_dict'])
        start_epoch = min(ck['epoch'] + 1, EPOCHS)
        best_score  = ck.get('best_score', best_score)
        print(f"[Trial {trial_idx}] Resuming from epoch {start_epoch}, prev best={best_score:.4f}")

    # 에폭 루프
    for epoch in range(start_epoch, EPOCHS+1):
        # ── Train ───────────────────────────────────────────────
        model.train()
        all_f_tr, all_a_tr, all_b_tr = [], [], []
        all_lbl_f_tr, all_lbl_a_tr, all_lbl_b_tr = [], [], []
        all_ids_tr = []

        for frames, pf_logits, f_lbl, a_lbl, b_lbl, ids in tqdm(
            train_loader,
            desc=f"Trial {trial_idx}/{N_TRIALS} ▶ Train Ep{epoch}/{EPOCHS}"
        ):
            try:
                video = torch.stack(frames, dim=2).to(DEVICE)
                slow = video[:, :, ::4]  # 예시 subsample
                fast = video
                pf_logits = pf_logits.to(DEVICE)
                out_f, out_a, out_b = model([slow, fast], pf_logits)
                loss = (
                    criterion(out_f, f_lbl.to(DEVICE))
                  + criterion(out_a, a_lbl.to(DEVICE))
                  + criterion(out_b, b_lbl.to(DEVICE))
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(f"[OOM] Trial {trial_idx}, Ep {epoch} ▶ pruning…")
                    torch.cuda.empty_cache()
                    raise optuna.exceptions.TrialPruned()
                else:
                    raise

            all_f_tr.append(out_f.detach().cpu())
            all_a_tr.append(out_a.detach().cpu())
            all_b_tr.append(out_b.detach().cpu())
            all_lbl_f_tr.append(f_lbl)
            all_lbl_a_tr.append(a_lbl)
            all_lbl_b_tr.append(b_lbl)
            all_ids_tr.extend(ids)

        scheduler.step()

        # train 정확도 계산 & 출력
        f_acc_tr = (torch.cat(all_f_tr).argmax(1) == torch.cat(all_lbl_f_tr)).float().mean().item()
        a_acc_tr = (torch.cat(all_a_tr).argmax(1) == torch.cat(all_lbl_a_tr)).float().mean().item()
        b_acc_tr = (torch.cat(all_b_tr).argmax(1) == torch.cat(all_lbl_b_tr)).float().mean().item()
        print(f"[Trial {trial_idx}/{N_TRIALS}] Ep{epoch}/{EPOCHS} ▶ "
              f"Train ▶ Feat Acc: {f_acc_tr:.4f}, A Acc: {a_acc_tr:.4f}, B Acc: {b_acc_tr:.4f}")

        # ── Validation ──────────────────────────────────────────
        model.eval()
        all_f_vl, all_a_vl, all_b_vl = [], [], []
        all_ids_vl = []
        v_corr_f = v_corr_a = v_corr_b = total = 0

        with torch.no_grad():
            for frames, pf_logits, f_lbl, a_lbl, b_lbl, ids in tqdm(
                val_loader,
                desc=f"Trial {trial_idx}/{N_TRIALS} ▶ Val   Ep{epoch}/{EPOCHS}"
            ):
                try:
                    video = torch.stack(frames, dim=2).to(DEVICE)
                    slow = video[:, :, ::4]  # 예시 subsample
                    fast = video
                    pf_logits = pf_logits.to(DEVICE)
                    out_f, out_a, out_b = model([slow, fast], pf_logits)
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print(f"[OOM] Trial {trial_idx}, Ep {epoch} ▶ pruning during val…")
                        torch.cuda.empty_cache()
                        raise optuna.exceptions.TrialPruned()
                    else:
                        raise

                v_corr_f += (out_f.argmax(1) == f_lbl.to(DEVICE)).sum().item()
                v_corr_a += (out_a.argmax(1) == a_lbl.to(DEVICE)).sum().item()
                v_corr_b += (out_b.argmax(1) == b_lbl.to(DEVICE)).sum().item()
                total  += f_lbl.size(0)
                all_f_vl.append(out_f.cpu())
                all_a_vl.append(out_a.cpu())
                all_b_vl.append(out_b.cpu())
                all_ids_vl.extend(ids)

        f_acc_vl = v_corr_f / total
        a_acc_vl = v_corr_a / total
        b_acc_vl = v_corr_b / total
        score    = 0.3 * f_acc_vl + 0.35 * a_acc_vl + 0.35 * b_acc_vl
        print(f"[Trial {trial_idx}/{N_TRIALS}] Ep{epoch}/{EPOCHS} ▶ "
              f"Val   ▶ Feat Acc: {f_acc_vl:.4f}, A Acc: {a_acc_vl:.4f}, B Acc: {b_acc_vl:.4f}, Score: {score:.4f}")

        # 체크포인트에 epoch, best_score 저장
        torch.save({
            'epoch': epoch,
            'best_score': best_score,
            'model_state_dict':    model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'scheduler_state_dict':scheduler.state_dict(),
        }, ckpt_path)

        # best 갱신 시 preds 저장
        if score > best_score:
            best_score = score
            save_preds(epoch, 'train', all_ids_tr, all_f_tr, all_a_tr, all_b_tr, preds_dir)
            save_preds(epoch,   'val', all_ids_vl, all_f_vl, all_a_vl, all_b_vl, preds_dir)

        # pruning
        trial.report(score, epoch-1)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return f_acc_vl, a_acc_vl, b_acc_vl

def objective(trial):
    hparams = {
        "lr":             trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay":   trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        "dropout":        trial.suggest_float("dropout", 0.3, 0.7),
        "optimizer_name": trial.suggest_categorical("optimizer_name", ["SGD","AdamW"]),
        "scheduler_name": trial.suggest_categorical("scheduler_name", ["StepLR","CosineAnnealingLR"]),
        "step_size":      trial.suggest_int("step_size", 5, 15),
        "gamma":          trial.suggest_float("gamma", 0.1, 0.9),
        "batch_size":     trial.suggest_categorical("batch_size", [8,16]),
        "focal_alpha":    trial.suggest_float("focal_alpha", 0.1, 0.5),
        "focal_gamma":    trial.suggest_float("focal_gamma", 1.0, 4.0),
    }
    f_acc, a_acc, b_acc = run_one_trial(hparams, trial)
    trial.set_user_attr("val_feat_acc", f_acc)
    trial.set_user_attr("val_a_acc",    a_acc)
    trial.set_user_attr("val_b_acc",    b_acc)
    return 0.3*f_acc + 0.35*a_acc + 0.35*b_acc

if __name__ == '__main__':
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(
        study_name="stage2_slowfast50",
        storage="sqlite:///stage2_slowfast50.db",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        pruner=pruner,
    )
    study.optimize(objective,
                   callbacks=[save_trials_to_csv, stop_after_n_complete])

    best = study.best_trial
    print(f"\nBest trial #{best.number}: score={best.value:.4f}")
    print("Params:", best.params)
