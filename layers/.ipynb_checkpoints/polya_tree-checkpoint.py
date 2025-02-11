import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
from typing import Optional, List

class LocalExpert(nn.Module):
    """
    - MLP classifier to be used in the leaf node
    - x : (batch_size, D),
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
    Feature selection by Gumbel-Sigmoid approximation to Beta-Bernoulli
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
                       (self.beta - 1) * torch.log(1 - phi + 1e-8)).mean() # sum()
        return prior_nll


class SoftPolyaTreeNode(nn.Module):
    """
    - The internal node determines the left/right child by gating (sigmoid).
    - leaf node => SoftBetaBernoulliFeatureSelector + LocalExpert
    """
    def __init__(self, dim_input, num_classes, max_depth=2, depth=0,
                 hidden_dim_expert=32, alpha_fs=1.0, beta_fs=1.0, gating_mlp_hidden = 16,
                 use_gating_mlp=False):
        super().__init__()
        self.dim_input = dim_input
        self.num_classes = num_classes
        self.max_depth = max_depth
        self.depth = depth
        self.is_leaf = (depth == max_depth)

        if use_gating_mlp:
            self.gating = nn.Sequential(
                nn.Linear(dim_input, gating_mlp_hidden),
                nn.Tanh(),
                nn.Linear(gating_mlp_hidden, 1)
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
        """
        - All nodes return (batch_size, num_classes) probability distribution
        """
        if self.is_leaf:
            gamma, _ = self.feature_selector(temperature, self.training)
            logits = self.local_expert(x, gamma_mask=gamma)
            probs = F.softmax(logits, dim=-1)  # In leaf, convert to probability
            return probs
        else:
            # Soft gating
            if self.gating is not None:
                gate_logit = self.gating(x).squeeze(-1)
            else:
                gate_logit = x @ self.w + self.b

            p_left = torch.sigmoid(gate_logit)  # (batch_size,)
            probs_left = self.left_child(x, temperature)
            probs_right = self.right_child(x, temperature)

            # probability mixture
            out = p_left.unsqueeze(1)*probs_left + (1 - p_left).unsqueeze(1)*probs_right
            return out
            
    def forward_hard(self, x):
        """ WE USE THIS FUNCTION AT TEST PERIOD
        - Interpretation: If p_left>0.5 instead of Soft Gating, it is 'hard' branching to the left or right.
        - Finally, masking is performed based on gamma>0.5 even in the leaf.
        """
        if self.is_leaf:
            # If you call it with 'training=False' in the feature selector, gamma = (phi>0.5)
            gamma, _ = self.feature_selector(temperature=1.0, training=False)
            logits = self.local_expert(x, gamma)
            probs = F.softmax(logits, dim=-1)
            return probs
        else:
            if self.gating is not None:
                gate_logit = self.gating(x).squeeze(-1)
            else:
                gate_logit = x @ self.w + self.b

            p_left = torch.sigmoid(gate_logit)
            # hard routing
            mask_left = (p_left > 0.5).float().unsqueeze(1)
            mask_right = 1 - mask_left
            # forward to child
            probs_left = self.left_child.forward_hard(x)
            probs_right = self.right_child.forward_hard(x)
            out = mask_left*probs_left + mask_right*probs_right
            return out
            
    def regularization_loss(self):
        reg_loss = 0.0
        weight_decay = 1e-4
        # Gating parameter regularization
        if self.gating is not None:
            for param in self.gating.parameters():
                reg_loss += weight_decay * (param**2).sum()
        else:
            if self.w is not None:
                reg_loss += weight_decay * (self.w**2).sum()
            if self.b is not None:
                reg_loss += weight_decay * (self.b**2).sum()

        if self.is_leaf:
            for param in self.local_expert.parameters():
                reg_loss += weight_decay * (param**2).sum()
            reg_loss += self.feature_selector.regularization_loss()
        else:
            reg_loss += self.left_child.regularization_loss()
            reg_loss += self.right_child.regularization_loss()

        return reg_loss