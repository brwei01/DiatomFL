import torch
import torch.nn as nn
import numpy as np
from scipy.special import lambertw

# 1. 将模型参数拼成向量
def model_to_vector(model):
    """
    将模型所有参数拼成一个1D向量
    """
    return torch.cat([p.data.view(-1) for p in model.parameters()])

# 2. Model Encoder（Autoencoder）
class ModelEncoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=16, use_small_arch=False):
        super().__init__()
        
        if use_small_arch or input_dim > 1000000:  # 如果参数太多，使用小架构
            # 小架构：直接映射到bottleneck
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, bottleneck_dim),
                nn.ReLU()
            )
            self.decoder = nn.Sequential(
                nn.Linear(bottleneck_dim, input_dim)
            )
        else:
            # 原始架构
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, bottleneck_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(bottleneck_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, input_dim)
            )

    def forward(self, theta):
        phi = self.encoder(theta)
        theta_hat = self.decoder(phi)
        return phi, theta_hat

# 3. MSE loss for autoencoder训练
def mse_loss(theta, theta_hat):
    return ((theta - theta_hat) ** 2).mean()

# 4. Indicator Function和最优sigma计算 $$$
def indicator_function(l_i, sigma_i, lambd, tau):
    """
    I_lambda(l_i, sigma_i) = (l_i - tau) * sigma_i + lambda * (log(sigma_i))^2
    """
    return (l_i - tau) * sigma_i + lambd * (np.log(sigma_i)) ** 2

def optimal_sigma(l_i, tau, lambd):
    """
    sigma*_i(l_i) = exp(-W(1/(2lambda) * max(-2/e, l_i - tau)))
    """
    x = max(-2/np.e, l_i - tau)
    w = lambertw(x / (2 * lambd)).real
    sigma_star = np.exp(-w)
    return sigma_star

# 5. 计算权重（用于拼接）$$$
def sample_weight(l_i, tau, lambd):
    sigma_star = optimal_sigma(l_i, tau, lambd)
    return 1.0 / sigma_star

# 6. 拼接高维meta-model特征和Indicator Function
def concat_meta_indicator(P_phi_t, I_lambda):
    """
    P_phi_t: shape [proj_dim] (meta model高维特征)
    I_lambda: float or shape [1] (indicator function)
    return: shape [proj_dim+1]
    """
    P_phi_t = torch.as_tensor(P_phi_t).flatten()
    I_lambda = torch.as_tensor([I_lambda]).flatten()
    return torch.cat([P_phi_t, I_lambda], dim=0)
    
def concat_meta_indicator_batch(P_phi_t, I_lambda_batch):
    """
    P_phi_t: shape [proj_dim] (单个meta model高维特征，对所有样本相同)
    I_lambda_batch: shape [batch] (每个样本的indicator function值)
    return: shape [batch, proj_dim+1]
    """
    batch_size = I_lambda_batch.shape[0]
    # 将P_phi_t广播到batch维度
    P_phi_t_batch = P_phi_t.unsqueeze(0).expand(batch_size, -1)  # [batch, proj_dim]
    I_lambda_batch = I_lambda_batch.unsqueeze(1)  # [batch, 1]
    return torch.cat([P_phi_t_batch, I_lambda_batch], dim=1)

# 7. meta model高维特征投影P(ϕt)
class MetaProjector(nn.Module):
    def __init__(self, input_dim, proj_dim=64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
    
    def forward(self, phi_t):
        return self.proj(phi_t)
    
    def forward_with_fallback(self, phi_t):
        """
        如果projector未训练或不可用，直接返回phi_t
        """
        try:
            return self.forward(phi_t)
        except:
            return phi_t

# 8. 经验值设置 $$$
def get_default_tau_lambda():
    # tau: 可设为0.5分位数或均值，lambda: 0.1~1之间
    tau = 0.5  # 可根据实际loss分布动态调整
    lambd = 0.5
    return tau, lambd 