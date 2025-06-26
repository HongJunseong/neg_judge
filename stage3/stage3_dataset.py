import torch
import pandas as pd
from torch.utils.data import Dataset
import torch.nn.functional as F

def softmax_with_temp(logits, T=1.5):
        return F.softmax(logits/T, dim=0)

class Stage3Dataset(Dataset):
    def __init__(self, pf_path, second_path, label_csv_path):
        pf = torch.load(pf_path)
        second = torch.load(second_path)

        self.pf_ids = pf["keys"]
        self.pf_logits = pf["logits"]

        self.second_ids = second["ids"]
        self.feat_logits = second["feat_logits"]
        self.a_logits = second["a_logits"]
        self.b_logits = second["b_logits"]

        # CSV 라벨 불러오기
        label_df = pd.read_csv(label_csv_path).set_index("video_id")
        self.label_map = {
            sid: int(label_df.loc[sid, "negligence_class"])  # 이미 클래스 0~10이면 그대로 int 변환만
            for sid in label_df.index
        }

        # 공통 sample_id만 추출
        self.sample_ids = [
            sid for sid in self.pf_ids if sid in self.second_ids and sid in self.label_map
        ]
        print(f"[Stage3Dataset] 총 샘플 수: {len(self.sample_ids)}")

        # 빠른 index 매핑
        self.pf_idx = {sid: i for i, sid in enumerate(self.pf_ids)}
        self.second_idx = {sid: i for i, sid in enumerate(self.second_ids)}

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]

        pf_logits = torch.tensor(self.pf_logits[self.pf_idx[sid]], dtype=torch.float32)
        feat_logits = torch.tensor(self.feat_logits[self.second_idx[sid]], dtype=torch.float32)
        a_logits = torch.tensor(self.a_logits[self.second_idx[sid]], dtype=torch.float32)
        b_logits = torch.tensor(self.b_logits[self.second_idx[sid]], dtype=torch.float32)

        # softmax 보정
        pf = softmax_with_temp(pf_logits, T=1.5)
        feat = softmax_with_temp(feat_logits, T=2.5)
        a = softmax_with_temp(a_logits, T=2.5)
        b = softmax_with_temp(b_logits, T=2.5)

        label = torch.tensor(self.label_map[sid], dtype=torch.long)
        return pf, feat, a, b, label
