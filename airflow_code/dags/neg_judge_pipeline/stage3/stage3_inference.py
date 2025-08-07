import torch
import numpy as np
from .stage3_model import Stage3NegligenceClassifier
from .stage3_dataset import softmax_with_temp  # 온도 스케일링 함수

def run_stage3_inference(pf_preds_path, stage2_preds_path, stage3_model_path):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    class_names = [f"{i*10}:{100-i*10}" for i in range(11)]

    # Load stage1 + stage2 logits
    pf_data = torch.load(pf_preds_path, map_location=DEVICE)
    stage2_data = torch.load(stage2_preds_path, map_location=DEVICE)

    video_id = pf_data["keys"][0]  # 단일 샘플 추론
    pf_logits = torch.tensor(pf_data["logits"][0], dtype=torch.float32)
    feat_logits = torch.tensor(stage2_data["feat_logits"][0], dtype=torch.float32)
    a_logits = torch.tensor(stage2_data["a_logits"][0], dtype=torch.float32)
    b_logits = torch.tensor(stage2_data["b_logits"][0], dtype=torch.float32)

    # Softmax 보정
    pf = softmax_with_temp(pf_logits, T=1.5).unsqueeze(0).to(DEVICE)
    feat = softmax_with_temp(feat_logits, T=2.5).unsqueeze(0).to(DEVICE)
    a = softmax_with_temp(a_logits, T=2.5).unsqueeze(0).to(DEVICE)
    b = softmax_with_temp(b_logits, T=2.5).unsqueeze(0).to(DEVICE)

    # Load model
    model = Stage3NegligenceClassifier(num_classes=11).to(DEVICE)
    ckpt = torch.load(stage3_model_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        logits = model(pf, feat, a, b)  # [1, 11]
        pred_class = logits.argmax(dim=1).item()
        pred_ratio = class_names[pred_class]

    print(f"✅ [최종 결과] 영상 ID: {video_id}")
    print(f"   🔹 예측 클래스: {pred_class}")
    print(f"   🔹 예상 과실 비율: {pred_ratio}")

    return {
        "video_id": video_id,
        "pred_class": pred_class,
        "pred_ratio": pred_ratio
    }
