import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
from typing import Optional, List

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



#class SoftBetaBernoulliFeatureSelector(nn.Module):
#    def __init__(self,
#                 num_features: int,
#                 alpha: float = 1.0,
#                 beta: float = 1.0,
#                 init_logit: Optional[torch.Tensor] = None,
#                 temperature: float = 0.1):
#        super().__init__()
#        
#        self.num_features = num_features
#        self.alpha = alpha
#        self.beta = beta
#        self.temperature = temperature  # Gumbel-Sigmoid 온도 파라미터
#        
#        if init_logit is None:
#            init_logit = torch.zeros(num_features)
#        self.logit_phi = nn.Parameter(init_logit)
#    
#    def forward(self, training=True):
#        phi = torch.sigmoid(self.logit_phi)  # (num_features,)
#        
#        if self.training and training:
#            u = torch.rand_like(phi)
#            g = -torch.log(-torch.log(u + 1e-10) + 1e-10)
#            gamma = torch.sigmoid((torch.log(phi + 1e-10) - torch.log(1-phi + 1e-10) + g) / self.temperature)
#        else:
#            gamma = (phi > 0.5).float()
#            
#        return gamma, phi
#    
#    def regularization_loss(self):
#        """
#        Beta(alpha, beta) 사전분포에 대해,
#        -log p(phi) ~ - [ (alpha-1)*log(phi) + (beta-1)*log(1-phi) ] 합
#        """
#        phi = torch.sigmoid(self.logit_phi)
#        prior_nll = - ((self.alpha - 1) * torch.log(phi + 1e-8) + \
#                       (self.beta - 1) * torch.log(1 - phi + 1e-8)).sum()
#        return prior_nll

