import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, 
                 reduction: str = 'mean'):
        """
        alpha: 클래스 불균형 시 소수 클래스에 부여할 가중치
        gamma: hard‐example에 집중할 정도
        reduction: 'none' | 'mean' | 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs: [B, C], raw logits
        # targets: [B] 정수 레이블
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')  # [B]
        p_t = torch.exp(-ce_loss)                                      # [B]
        focal_term = (1 - p_t) ** self.gamma                           # [B]
        loss = self.alpha * focal_term * ce_loss                       # [B]

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss  # 'none'
