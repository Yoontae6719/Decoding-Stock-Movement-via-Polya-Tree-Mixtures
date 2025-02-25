import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
from typing import Optional, List
from layers.polya_tree import SoftPolyaTreeNode

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.root = SoftPolyaTreeNode(
            dim_input   = configs.dim_input,
            num_classes = configs.num_classes,
            max_depth   = configs.max_depth,
            depth = 0,
            hidden_dim_expert = configs.hidden_dim_expert,
            alpha_fs    = configs.alpha_fs,
            beta_fs     = configs.beta_fs,
            use_gating_mlp = configs.use_gating_mlp
        )
        self.dim_input = configs.dim_input
        self.num_classes = configs.num_classes
        self.max_depth = configs.max_depth
    
    def forward(self, x, temperature=0.5):
        probs = self.root(x, temperature)
        return probs
    
    def forward_hard(self, x):
        # for xai (test period): hard gating + hard mask
        return self.root.forward_hard(x)