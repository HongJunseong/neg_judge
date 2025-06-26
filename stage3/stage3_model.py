import torch
import torch.nn as nn

class Stage3NegligenceClassifier(nn.Module):
    def __init__(self,
                 num_pf=10,
                 num_feat=35,
                 num_a=60,
                 num_b=61,
                 num_classes=11,    # 과실비율을 n개 구간으로 분류할 경우
                 hidden_dim=1024,
                 dropout=0.7):
        super().__init__()

        self.input_dim = num_pf + num_feat + num_a + num_b

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),

            nn.Linear(hidden_dim // 2, num_classes)  # <-- 분류
        )

    def forward(self, pf_logits, feat_logits, a_logits, b_logits):
        x = torch.cat([pf_logits, feat_logits, a_logits, b_logits], dim=1)
        return self.model(x)  # shape: [B, num_classes]
