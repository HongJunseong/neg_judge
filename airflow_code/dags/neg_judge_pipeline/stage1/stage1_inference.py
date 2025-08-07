import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from .stage1_model import MultiFrameClassifier
from .stage1_dataset import MultiFrameDataset
from .stage1_config import BATCH_SIZE, FRAMES_PER_SAMPLE, NUM_CLASSES


def run_stage1_inference(frame_folder: str, output_path: str, checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert os.path.exists(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"
    assert os.path.exists(frame_folder), f"Frame folder not found: {frame_folder}"

    # preprocessing
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    # Dataset & Loader
    dataset = MultiFrameDataset(
        image_root=frame_folder,
        transform=test_transform,
        frames_per_sample=FRAMES_PER_SAMPLE
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # model load
    model = MultiFrameClassifier(NUM_CLASSES, FRAMES_PER_SAMPLE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()

    # inference
    all_logits = []
    all_keys = []

    with torch.no_grad():
        for frames, video_ids in dataloader:
            # [batch, frames, C, H, W] 형식으로 변환 필요 시 수정
            frames = [f.to(device) for f in frames]  # frames: list of tensors
            logits = model(frames)
            all_logits.append(logits.cpu())
            all_keys.extend(video_ids)

    result = {
        "keys": all_keys,
        "logits": torch.cat(all_logits, dim=0)
    }

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    torch.save(result, output_path)
    print(f"[Stage1 추론 결과 저장 완료] -> {output_path}")

    # 가장 높은 확률 클래스 출력
    top1 = result["logits"].argmax(dim=1).item()
    print(f"예측된 사고 장소 클래스: {top1}")
    return top1
