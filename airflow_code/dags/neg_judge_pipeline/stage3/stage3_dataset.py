# stage3_dataset.py

import torch
import pandas as pd
from torch.utils.data import Dataset
import torch.nn.functional as F

def softmax_with_temp(logits, T=1.5):
    return F.softmax(logits / T, dim=0)

class Stage3Dataset(Dataset):
    def __init__(self, pf_path, second_path):
        pf = torch.load(pf_path)
        second = torch.load(second_path)

        self.pf_ids = pf["keys"]
        self.pf_logits = pf["logits"]

        self.second_ids = second["ids"]
        self.feat_logits = second["feat_logits"]
        self.a_logits = second["a_logits"]
        self.b_logits = second["b_logits"]

        # 공통 sample_id만 추출
        self.sample_ids = [sid for sid in self.pf_ids if sid in self.second_ids]
        print(f"[Stage3Dataset: 추론용] 총 샘플 수: {len(self.sample_ids)}")

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

        pf   = softmax_with_temp(pf_logits, T=1.5)
        feat = softmax_with_temp(feat_logits, T=2.5)
        a    = softmax_with_temp(a_logits, T=2.5)
        b    = softmax_with_temp(b_logits, T=2.5)

        return pf, feat, a, b, sid  # ← 라벨 제거하고 sid 반환