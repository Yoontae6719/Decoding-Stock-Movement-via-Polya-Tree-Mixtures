import torch
import torch.nn as nn
import torch.nn.functional as F


def kl_bernoulli_beta(phi: torch.Tensor, alpha: float, beta: float, eps=1e-8) -> torch.Tensor:
    phi = phi.clamp(eps, 1 - eps)

    c = (
        torch.lgamma(torch.tensor(alpha, device=phi.device)) +
        torch.lgamma(torch.tensor(beta, device=phi.device)) -
        torch.lgamma(torch.tensor(alpha + beta, device=phi.device))
    )
    #   - (alpha - 1)*log(phi) - (beta - 1)*log(1-phi)
    log_term = (alpha - 1) * phi.log() + (beta - 1) * (1 - phi).log()

    kl =c - log_term
    return kl



class BetaBernoulliFeatureSelector(nn.Module):
    def __init__(self,
                 num_features: int,
                 alpha: float = 1.0,
                 beta: float = 1.0,
                 init_logit_phi: float = 0.0):
        super().__init__()
        self.num_features = num_features
        self.alpha = alpha
        self.beta = beta
        self.logit_phi = nn.Parameter(torch.full((num_features,), init_logit_phi))
    
    def forward(self, temperature: float = 0.5, training=True):
        phi = torch.sigmoid(self.logit_phi)
        if self.training and training:
            eps = 1e-10
            u = torch.rand_like(phi)
            g = -torch.log(-torch.log(u + eps))
            
            gamma = torch.sigmoid( (torch.log(phi + eps) - torch.log(1 - phi + eps) + g) / temperature )
        else:
            # hard threshold (Note that we do not use this term)
            gamma = (phi > 0.5).float()
        
        return gamma, phi
    
    def get_kl_regularizer(self) -> torch.Tensor:
        phi = torch.sigmoid(self.logit_phi)          # (num_features,)
        kl_vals = kl_bernoulli_beta(phi, self.alpha, self.beta)  # (num_features,)
        return kl_vals.sum()



class LocalExpert(nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(num_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h)


class BetaBernoulliPolyaTreeNode(nn.Module):
    def __init__(self,
                 dim_input: int,
                 num_classes: int,
                 max_depth: int,
                 depth: int = 0,
                 hidden_dim_expert: int = 32,
                 alpha_fs: float = 1.0,
                 beta_fs: float = 1.0,
                 use_feature_selection: bool = True,
                 use_leaf_feature_selector_only: bool = False,
                 use_gating_mlp: bool = False,
                 gating_mlp_hidden: int = 16):
        super().__init__()
        self.dim_input = dim_input
        self.num_classes = num_classes
        self.max_depth = max_depth
        self.depth = depth
        
        self.is_leaf = (depth == max_depth)
        
        if use_leaf_feature_selector_only and not self.is_leaf:
            self.use_feature_selection = False
        else:
            self.use_feature_selection = use_feature_selection
        
        if self.use_feature_selection:
            self.fs_module = BetaBernoulliFeatureSelector(
                num_features = dim_input,
                alpha = alpha_fs,
                beta  = beta_fs,
                init_logit_phi = 0.0
            )
        else:
            self.fs_module = None
        
        if self.is_leaf:
            self.local_expert = LocalExpert(dim_input, num_classes, hidden_dim_expert)
            self.left_child = None
            self.right_child = None
        else:
            if use_gating_mlp:
                self.gating = nn.Sequential(
                    nn.Linear(dim_input, gating_mlp_hidden),
                    nn.Tanh(),
                    nn.Linear(gating_mlp_hidden, 1)
                )
                self.w = None
                self.b = None
            else:
                self.gating = None
                self.w = nn.Parameter(torch.zeros(dim_input))
                self.b = nn.Parameter(torch.zeros(1))
            
            self.left_child = BetaBernoulliPolyaTreeNode(
                dim_input, num_classes, max_depth,
                depth=depth+1,
                hidden_dim_expert=hidden_dim_expert,
                alpha_fs=alpha_fs,
                beta_fs=beta_fs,
                use_feature_selection=use_feature_selection,
                use_leaf_feature_selector_only=use_leaf_feature_selector_only,
                use_gating_mlp=use_gating_mlp,
                gating_mlp_hidden=gating_mlp_hidden
            )
            self.right_child = BetaBernoulliPolyaTreeNode(
                dim_input, num_classes, max_depth,
                depth=depth+1,
                hidden_dim_expert=hidden_dim_expert,
                alpha_fs=alpha_fs,
                beta_fs=beta_fs,
                use_feature_selection=use_feature_selection,
                use_leaf_feature_selector_only=use_leaf_feature_selector_only,
                use_gating_mlp=use_gating_mlp,
                gating_mlp_hidden=gating_mlp_hidden
            )
    
    def forward(self, x: torch.Tensor,
                temperature_fs: float = 0.5,
                temperature_gate: float = 0.5) -> torch.Tensor:

        if self.use_feature_selection:
            gamma, _ = self.fs_module(temperature=temperature_fs, training=self.training)
            x_masked = x * gamma.unsqueeze(0)  # (batch_size, dim_input)
        else:
            x_masked = x
        
        if self.is_leaf:
            logits = self.local_expert(x_masked)
            probs = F.softmax(logits, dim=-1)
            return probs
        else:
            if self.gating is not None:
                gate_logit = self.gating(x_masked).squeeze(-1)  # (batch_size,)
            else:
                gate_logit = (x_masked * self.w).sum(dim=1) + self.b  # (batch_size,)
            
            p_left = torch.sigmoid(gate_logit / temperature_gate)
            
            probs_left = self.left_child(x_masked, temperature_fs, temperature_gate)
            probs_right = self.right_child(x_masked, temperature_fs, temperature_gate)
            
            probs = p_left.unsqueeze(-1) * probs_left + (1 - p_left).unsqueeze(-1) * probs_right
            return probs
    
    def forward_hard(self, x: torch.Tensor, threshold_fs: float = 0.5) -> torch.Tensor:
        if self.use_feature_selection:
            with torch.no_grad():
                phi = torch.sigmoid(self.fs_module.logit_phi)
                gamma_hard = (phi > threshold_fs).float()
            x_masked = x * gamma_hard.unsqueeze(0)
        else:
            x_masked = x
        
        if self.is_leaf:
            logits = self.local_expert(x_masked)
            return F.softmax(logits, dim=-1)
        else:
            if self.gating is not None:
                gate_logit = self.gating(x_masked).squeeze(-1)
            else:
                gate_logit = (x_masked * self.w).sum(dim=1) + self.b
            
            p_left = torch.sigmoid(gate_logit)
            mask_left = (p_left > 0.5).float().unsqueeze(-1)
            mask_right = 1 - mask_left
            
            probs_left = self.left_child.forward_hard(x_masked, threshold_fs)
            probs_right = self.right_child.forward_hard(x_masked, threshold_fs)
            return mask_left * probs_left + mask_right * probs_right
    
    def get_kl_loss(self) -> torch.Tensor:

        device = torch.device("cpu")
        if self.use_feature_selection:
            device = self.fs_module.logit_phi.device
        elif self.w is not None:
            device = self.w.device
        
        kl_val = torch.zeros((), device=device)
        if self.use_feature_selection:
            kl_val = kl_val + self.fs_module.get_kl_regularizer()
        
        if not self.is_leaf:
            kl_val = kl_val + self.left_child.get_kl_loss()
            kl_val = kl_val + self.right_child.get_kl_loss()
        
        return kl_val


class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()
        
        self.root = BetaBernoulliPolyaTreeNode(
            dim_input = configs.dim_input,
            num_classes = configs.num_classes,
            max_depth = configs.max_depth,
            depth = 0,
            hidden_dim_expert = configs.hidden_dim_expert,
            alpha_fs = configs.alpha_fs,
            beta_fs = configs.beta_fs,
            use_feature_selection = configs.use_feature_selection,
            use_leaf_feature_selector_only = configs.use_leaf_feature_selector_only,
            use_gating_mlp = configs.use_gating_mlp,
            gating_mlp_hidden = configs.gating_mlp_hidden
        )
        self.dim_input = configs.dim_input
        self.num_classes = configs.num_classes
        self.max_depth = configs.max_depth
    
    def forward(self, x: torch.Tensor,
                temperature_fs: float = 0.5,
                temperature_gate: float = 0.5) -> torch.Tensor:
        return self.root(x, temperature_fs, temperature_gate)
    
    def forward_hard(self, x: torch.Tensor, threshold_fs: float = 0.5) -> torch.Tensor:
        return self.root.forward_hard(x, threshold_fs)
    
    def get_kl_loss(self) -> torch.Tensor:
        return self.root.get_kl_loss()
