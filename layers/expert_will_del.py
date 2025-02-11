import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
from typing import Optional, List

class LocalExpert(nn.Module):
    """
    - 리프 노드에서 사용될 간단한 MLP 분류기
    - 입력 x는 (batch_size, D),
      마스크 gamma는 (D,) 형태로 곱해준 뒤 MLP 처리
    """
    def __init__(self, num_features, num_classes, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(num_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x, gamma_mask):
        """
        x: (batch_size, D)
        gamma_mask: (D,) in (0,1), soft mask
        """
        x_masked = x * gamma_mask  # soft masking
        h = F.relu(self.fc1(x_masked))
        logits = self.fc2(h)  # (batch_size, num_classes)
        return logits
