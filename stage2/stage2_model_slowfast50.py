import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pytorchvideo.models.hub import slowfast_r50

class Stage2Model(nn.Module):
    """
    SlowFast-R50 기반 멀티태스크 모델:
    입력: 영상 [B, 3, T, H, W], 1차 예측 logits
    출력: 사고장소 특징, A 진행, B 진행
    """
    def __init__(self,
                 num_pf: int,
                 num_feat: int,
                 num_a: int,
                 num_b: int,
                 pf_emb_dim: int = 128,
                 head_hidden: int = 256,
                 dropout: float = 0.55,
                 temperature: float = 2.0):
        super().__init__()
        self.temperature = temperature

        # SlowFast-R50 백본 (pytorchvideo)
        self.backbone = slowfast_r50(pretrained=True)
        self.backbone.blocks[-1].proj = nn.Identity()
        self.feat_dim = 2304 # (2048 + 256)

        # 1차 logits 임베딩
        self.pf_emb = nn.Sequential(
            nn.Linear(num_pf, pf_emb_dim),
            nn.BatchNorm1d(pf_emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.39)
        )

        fusion_dim = self.feat_dim + pf_emb_dim

        def make_head(out_dim):
            return nn.Sequential(
                nn.Linear(fusion_dim, head_hidden),
                nn.BatchNorm1d(head_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, out_dim)
            )

        self.feat_head = make_head(num_feat)
        self.a_head    = make_head(num_a)
        self.b_head    = make_head(num_b)

    def forward(self, frames, pf_logits):
        # frames: [B, 3, T, H, W]
        # Slow: 1/4 subsampled, Fast: full
        slow, fast = frames

        # pytorchvideo는 list로 두 stream 입력
        features = self.backbone([slow, fast])  # [B, 2304]

        # 1차 예측 softmax + embedding
        pf_probs = F.softmax(pf_logits / self.temperature, dim=1).detach()
        pf_emb = self.pf_emb(pf_probs)

        x = torch.cat([features, pf_emb], dim=1)
        return self.feat_head(x), self.a_head(x), self.b_head(x)
