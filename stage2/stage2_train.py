import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torchvision import transforms
from torchvision.transforms import RandAugment, RandomErasing
from tqdm import tqdm
import pandas as pd
from stage2_dataset import Stage2Dataset  # __getitem__ returns (frames, pf_logits, f_lbl, a_lbl, b_lbl, sample_id)
from stage2_model_slowfast50 import Stage2Model
from stage2_config import *
from stage2_utils import FocalLoss


# 1차 모델 결과 디렉토리
CHECKPOINT_DIR = "./checkpoints4"
# 2차 모델 결과 및 preds 디렉토리
CHECKPOINT_DIR_2 = "./checkpoints6_result"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR_2, exist_ok=True)

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

best_weighted_score = 0.0

def save_preds(epoch, split, ids, feat_list, a_list, b_list):
    """Train/Val epoch별 logits 파일 저장"""
    filepath = os.path.join(CHECKPOINT_DIR_2, f"{split}_stage2_preds_ep{epoch}.pth")
    torch.save({
        'ids': ids,
        'feat_logits': torch.cat(feat_list, dim=0),
        'a_logits':    torch.cat(a_list,    dim=0),
        'b_logits':    torch.cat(b_list,    dim=0),
    }, filepath)
    print(f"[Saved {split.upper()} preds] {filepath}")

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_csv_path = os.path.join(CHECKPOINT_DIR_2, "stage2_train_log.csv")
    if os.path.exists(log_csv_path):
        df_existing = pd.read_csv(log_csv_path)
        # 마지막으로 기록된 epoch 번호
        last_logged_epoch = int(df_existing["epoch"].max())

        # 기존 로그를 리스트로 복원
        train_loss_list      = df_existing["train_loss"].tolist()
        train_f_acc_list     = df_existing["train_feat_acc"].tolist()
        train_a_acc_list     = df_existing["train_a_acc"].tolist()
        train_b_acc_list     = df_existing["train_b_acc"].tolist()
        val_loss_list        = df_existing["val_loss"].tolist()
        val_f_acc_list       = df_existing["val_feat_acc"].tolist()
        val_a_acc_list       = df_existing["val_a_acc"].tolist()
        val_b_acc_list       = df_existing["val_b_acc"].tolist()

        # 이어서 학습할 epoch 설정
        start_epoch = last_logged_epoch + 1
        print(f"[Resume Logs] Found existing CSV. Last_epoch={last_logged_epoch}, resuming from epoch {start_epoch}")
    else:
        # 처음 실행 시: 빈 리스트로 초기화, start_epoch=1
        train_loss_list      = []
        train_f_acc_list     = []
        train_a_acc_list     = []
        train_b_acc_list     = []
        val_loss_list        = []
        val_f_acc_list       = []
        val_a_acc_list       = []
        val_b_acc_list       = []
        start_epoch = 1
        print("[New Run] No existing CSV. Starting from epoch 1.")
    # ───────────────────────────────────────────

    # transforms 정의
    train_transform = transforms.Compose([
        #transforms.Resize((256, 455)),
        #transforms.RandomResizedCrop(256, scale=(0.65625, 1.0), ratio=(1.0,1.0)),
        #transforms.Resize((244, 244)),
        #transforms.RandomHorizontalFlip(p=0.5),
        #transforms.ToTensor(),
        #transforms.Normalize(mean, std),
        transforms.Resize(256),                      # 짧은 변 256px, 종횡비 유지
        transforms.RandomResizedCrop(224, scale=(0.8,1.0)),  
        transforms.RandomHorizontalFlip(p=0.5),
        RandAugment(num_ops=2, magnitude=9),         # 색·형태 변형
        transforms.ColorJitter(0.1,0.1,0.1,0.5), # 절반으로 #0.5 였음 끝에
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        RandomErasing(p=0.1, scale=(0.02,0.1)),     # 일부 지우기 # 0.3으로 했었음
    ])
    val_transform = transforms.Compose([
        #transforms.Resize((256, 455)),
        #transforms.CenterCrop((244, 244)),
        #transforms.ToTensor(),
        #transforms.Normalize(mean, std),
        transforms.Resize(256),        # 종횡비 유지
        transforms.CenterCrop(224),    # 중앙 224×224
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # 데이터셋 & 로더
    train_ds = Stage2Dataset(
        image_root='data/train',
        feature_txt='dataset_txt/train_accident_place_feature_mapped.txt',
        a_txt='dataset_txt/train_vehicle_a_progress_info_mapped.txt',
        b_txt='dataset_txt/train_vehicle_b_progress_info_mapped.txt',
        preds_path=os.path.join(CHECKPOINT_DIR, 'train_first_preds_ep43.pth'),
        transform=train_transform,
        random_sampling=True
    )
    val_ds = Stage2Dataset(
        image_root='data/val',
        feature_txt='dataset_txt/val_accident_place_feature_mapped.txt',
        a_txt='dataset_txt/val_vehicle_a_progress_info_mapped.txt',
        b_txt='dataset_txt/val_vehicle_b_progress_info_mapped.txt',
        preds_path=os.path.join(CHECKPOINT_DIR, 'val_first_preds_ep43.pth'),
        transform=val_transform,
        random_sampling=False
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 모델, 옵티마이저, 스케줄러, 손실함수
    model = Stage2Model(
        num_pf=NUM_CLASSES,
        num_feat=NUM_FEATURES,
        num_a=NUM_A_PROGRESS,
        num_b=NUM_B_PROGRESS
    ).to(device)

    #optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)

    #optimizer = torch.optim.SGD(model.parameters(), lr=0.0014, momentum=0.9, weight_decay=0.0009)
    optimizer = torch.optim.AdamW(model.parameters(), lr= 1e-4, weight_decay=7.45e-5)
    #scheduler = StepLR(optimizer, step_size=12, gamma=0.63)
    #criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = FocalLoss(alpha=0.23, gamma=1.07)


    # 체크포인트 로드 (마지막 상태만)
    start_epoch = 1
    ckpt_path = os.path.join(CHECKPOINT_DIR_2, 'stage2_last.pth')
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck['model_state_dict'])
        optimizer.load_state_dict(ck['optimizer_state_dict'])
        scheduler.load_state_dict(ck['scheduler_state_dict'])
        start_epoch = ck['epoch'] + 1
        print(f"[RESUME] from epoch {start_epoch}")

    # 학습 및 평가 루프
    for epoch in range(start_epoch, EPOCHS+1):
        # ----- Train -----
        model.train()
        train_loss, train_total = 0.0, 0
        all_f_tr, all_a_tr, all_b_tr = [], [], []
        all_lbl_f_tr, all_lbl_a_tr, all_lbl_b_tr = [], [], []
        all_ids_tr = []

        for frames, pf_logits, f_lbl, a_lbl, b_lbl, ids in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            video = torch.stack(frames, dim=2).to(device)
            pf_logits = pf_logits.to(device)
            out_f, out_a, out_b = model([video[:,:,::4], video], pf_logits)
            loss = (criterion(out_f, f_lbl.to(device)) +
                    criterion(out_a, a_lbl.to(device)) +
                    criterion(out_b, b_lbl.to(device)))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            train_loss += loss.item() * f_lbl.size(0)
            train_total += f_lbl.size(0)

            # logits & labels & ids 수집
            all_f_tr.append(out_f.detach().cpu())
            all_a_tr.append(out_a.detach().cpu())
            all_b_tr.append(out_b.detach().cpu())
            all_lbl_f_tr.append(f_lbl)
            all_lbl_a_tr.append(a_lbl)
            all_lbl_b_tr.append(b_lbl)
            all_ids_tr.extend(ids)


        # train 정확도 계산 및 출력
        f_acc_tr = (torch.cat(all_f_tr).argmax(1) == torch.cat(all_lbl_f_tr)).float().mean().item()
        a_acc_tr = (torch.cat(all_a_tr).argmax(1) == torch.cat(all_lbl_a_tr)).float().mean().item()
        b_acc_tr = (torch.cat(all_b_tr).argmax(1) == torch.cat(all_lbl_b_tr)).float().mean().item()
        avg_train_loss = train_loss / train_total

        print(f"Epoch {epoch} Train ▶ Loss: {avg_train_loss:.4f}, "
              f"Feat Acc: {f_acc_tr:.4f}, A Acc: {a_acc_tr:.4f}, B Acc: {b_acc_tr:.4f}")

        # train preds 저장
        save_preds(epoch, 'train', all_ids_tr, all_f_tr, all_a_tr, all_b_tr)

        # ----- Validation -----
        model.eval()
        all_f_vl, all_a_vl, all_b_vl = [], [], []
        all_lbl_f_vl, all_lbl_a_vl, all_lbl_b_vl = [], [], []
        all_ids_vl = []

        val_loss = 0.0  
        v_corr_f = v_corr_a = v_corr_b = 0
        v_total = 0

        with torch.no_grad():
            for frames, pf_logits, f_lbl, a_lbl, b_lbl, ids in tqdm(val_loader, desc=f"Val Epoch {epoch}"):
                video = torch.stack(frames, dim=2).to(device)
                pf_logits = pf_logits.to(device)
                out_f, out_a, out_b = model([video[:,:,::4], video], pf_logits)

                loss_val = (criterion(out_f, f_lbl.to(device))
                            + criterion(out_a, a_lbl.to(device))
                            + criterion(out_b, b_lbl.to(device)))
                val_loss += loss_val.item() * f_lbl.size(0)

                # 정확도 집계
                v_corr_f += (out_f.argmax(1) == f_lbl.to(device)).sum().item()
                v_corr_a += (out_a.argmax(1) == a_lbl.to(device)).sum().item()
                v_corr_b += (out_b.argmax(1) == b_lbl.to(device)).sum().item()
                v_total += f_lbl.size(0)

                # logits & labels & ids 수집
                all_f_vl.append(out_f.cpu())
                all_a_vl.append(out_a.cpu())
                all_b_vl.append(out_b.cpu())
                all_lbl_f_vl.append(f_lbl)
                all_lbl_a_vl.append(a_lbl)
                all_lbl_b_vl.append(b_lbl)
                all_ids_vl.extend(ids)

        avg_val_loss = val_loss / v_total        

        f_acc_vl = v_corr_f / v_total
        a_acc_vl = v_corr_a / v_total
        b_acc_vl = v_corr_b / v_total
        print(f"Epoch {epoch} Val   ▶ Loss: {avg_val_loss:.4f}, Feat Acc: {f_acc_vl:.4f}, A Acc: {a_acc_vl:.4f}, B Acc: {b_acc_vl:.4f}")

        # ---- Best 모델 갱신: 가중치 0.3 / 0.35 / 0.35 적용 ----
        weighted_score = 0.3 * f_acc_vl + 0.35 * a_acc_vl + 0.35 * b_acc_vl
        if weighted_score > best_weighted_score:
            best_weighted_score = weighted_score
            best_path = os.path.join(CHECKPOINT_DIR_2, "stage2_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'weighted_score': weighted_score
            }, best_path)
            print(f"👉 New best weighted model saved (epoch {epoch}, score {weighted_score:.4f}) at\n   {best_path}")

        # val preds 저장
        save_preds(epoch, 'val', all_ids_vl, all_f_vl, all_a_vl, all_b_vl)

         # ─── 로그 업데이트 및 CSV 덮어쓰기 ─────────────────────────────────────────
        train_loss_list.append(avg_train_loss)
        train_f_acc_list.append(f_acc_tr)
        train_a_acc_list.append(a_acc_tr)
        train_b_acc_list.append(b_acc_tr)
        val_loss_list.append(avg_val_loss)
        val_f_acc_list.append(f_acc_vl)
        val_a_acc_list.append(a_acc_vl)
        val_b_acc_list.append(b_acc_vl)

        # 전체 epoch 범위(1~현재) 로그로 DataFrame 재생성 → 덮어쓰기
        log_df = pd.DataFrame({
            "epoch": list(range(1, len(train_loss_list) + 1)),
            "train_loss": train_loss_list,
            "train_feat_acc": train_f_acc_list,
            "train_a_acc": train_a_acc_list,
            "train_b_acc": train_b_acc_list,
            "val_loss": val_loss_list, 
            "val_feat_acc": val_f_acc_list,
            "val_a_acc": val_a_acc_list,
            "val_b_acc": val_b_acc_list,
        })
        log_df.to_csv(log_csv_path, index=False)
        # ───────────────────────────────────────────────────────────────────────────

        # 마지막 체크포인트 저장
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, ckpt_path)
        print(f"[Saved last checkpoint] {ckpt_path}")

        scheduler.step()
