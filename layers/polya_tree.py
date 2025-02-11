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

class SoftBetaBernoulliFeatureSelector(nn.Module):
    """
    Beta-Bernoulli에 대한 Gumbel-Sigmoid 근사로 feature selection
    """
    def __init__(self, num_features, alpha=1.0, beta=1.0):
        super().__init__()
        self.num_features = num_features
        self.alpha = alpha
        self.beta = beta
        self.logit_phi = nn.Parameter(torch.zeros(num_features)) # init -> phi=0.5 근처

    def forward(self, temperature: float, training=True):
        phi = torch.sigmoid(self.logit_phi)
        if self.training and training:
            u = torch.rand_like(phi)
            g = -torch.log(-torch.log(u + 1e-10) + 1e-10)
            gamma = torch.sigmoid((torch.log(phi+1e-10) - torch.log(1-phi+1e-10) + g) / temperature)
        else:
            gamma = (phi > 0.5).float()
        return gamma, phi

    def regularization_loss(self):
        phi = torch.sigmoid(self.logit_phi)
        prior_nll = - ((self.alpha - 1) * torch.log(phi + 1e-8) +
                       (self.beta - 1) * torch.log(1 - phi + 1e-8)).sum()
        return prior_nll


class SoftPolyaTreeNode(nn.Module):
    """
    - 내부 노드는 gating(시그모이드)으로 왼/오른쪽 child 결정
    - leaf 노드는 SoftBetaBernoulliFeatureSelector + LocalExpert
    """
    def __init__(self, dim_input, num_classes, max_depth=2, depth=0,
                 hidden_dim_expert=32, alpha_fs=1.0, beta_fs=1.0,
                 use_gating_mlp=False):
        super().__init__()
        self.dim_input = dim_input
        self.num_classes = num_classes
        self.max_depth = max_depth
        self.depth = depth
        self.is_leaf = (depth == max_depth)

        if use_gating_mlp:
            self.gating = nn.Sequential(
                nn.Linear(dim_input, 16),
                nn.Tanh(),
                nn.Linear(16, 1)
            )
            self.w, self.b = None, None
        else:
            self.gating = None
            self.w = nn.Parameter(torch.randn(dim_input)*0.1)
            self.b = nn.Parameter(torch.zeros(1))

        if self.is_leaf:
            self.feature_selector = SoftBetaBernoulliFeatureSelector(dim_input, alpha_fs, beta_fs)
            self.local_expert = LocalExpert(dim_input, num_classes, hidden_dim_expert)
            self.left_child = None
            self.right_child = None
        else:
            self.left_child = SoftPolyaTreeNode(
                dim_input, num_classes, max_depth, depth+1,
                hidden_dim_expert, alpha_fs, beta_fs, use_gating_mlp
            )
            self.right_child = SoftPolyaTreeNode(
                dim_input, num_classes, max_depth, depth+1,
                hidden_dim_expert, alpha_fs, beta_fs, use_gating_mlp
            )

    def forward(self, x, temperature=0.5):
        if self.is_leaf:
            gamma, _ = self.feature_selector(temperature, self.training)
            logits = self.local_expert(x, gamma_mask=gamma)
            return logits
        else:
            if self.gating is not None:
                gate_logit = self.gating(x).squeeze(-1)
            else:
                gate_logit = x @ self.w + self.b
            p_left = torch.sigmoid(gate_logit)
            out_left = self.left_child(x, temperature)
            out_right = self.right_child(x, temperature)
            p_left = p_left.unsqueeze(1)
            out = p_left*out_left + (1-p_left)*out_right
            return out

    def regularization_loss(self):
        reg_loss = 0.0
        weight_decay = 1e-4
        # 게이팅 파라미터에 L2
        if self.gating is not None:
            for param in self.gating.parameters():
                reg_loss += weight_decay * (param**2).sum()
        else:
            if self.w is not None:
                reg_loss += weight_decay * (self.w**2).sum()
            if self.b is not None:
                reg_loss += weight_decay * (self.b**2).sum()

        if self.is_leaf:
            reg_loss += self.feature_selector.regularization_loss()
        else:
            reg_loss += self.left_child.regularization_loss()
            reg_loss += self.right_child.regularization_loss()
        return reg_loss