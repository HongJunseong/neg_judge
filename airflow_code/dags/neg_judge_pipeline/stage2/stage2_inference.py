import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from .stage2_dataset import Stage2Dataset
from .stage2_model_slowfast50 import Stage2Model
from .stage2_config import BATCH_SIZE, NUM_CLASSES, NUM_FEATURES, NUM_A_PROGRESS, NUM_B_PROGRESS

def run_stage2_inference(frame_folder, stage1_preds_path, stage2_model_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Stage1에서 예측된 logits 불러오기
    pf_data = torch.load(stage1_preds_path, map_location=device)
    pf_logits = pf_data["logits"]  # [1, NUM_CLASSES]
    keys = pf_data["keys"]         # e.g., ['car_accident']

    video_id = os.path.basename(frame_folder)  # 단일 샘플 추론

    # preprocessing
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Dataset & DataLoader
    dataset = Stage2Dataset(
        image_root=os.path.dirname(frame_folder),
        video_ids=[video_id],
        pf_logits=pf_logits,
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # model load
    model = Stage2Model(
        num_pf=NUM_CLASSES,
        num_feat=NUM_FEATURES,
        num_a=NUM_A_PROGRESS,
        num_b=NUM_B_PROGRESS
    ).to(device)
    ckpt = torch.load(stage2_model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # inference
    all_feat_logits, all_a_logits, all_b_logits = [], [], []

    with torch.no_grad():
        for frames, pf_input, ids in dataloader:
            video = torch.stack(frames, dim=2).to(device)
            pf_input = pf_input.to(device)
            out_f, out_a, out_b = model([video[:,:,::4], video], pf_input)
            all_feat_logits.append(out_f.cpu())
            all_a_logits.append(out_a.cpu())
            all_b_logits.append(out_b.cpu())

    # save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result = {
        "ids": keys,
        "feat_logits": torch.cat(all_feat_logits),
        "a_logits": torch.cat(all_a_logits),
        "b_logits": torch.cat(all_b_logits)
    }
    torch.save(result, output_path)
    print(f"[Stage2 추론 결과 저장 완료] → {output_path}")
