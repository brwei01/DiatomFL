import torch
import torch.nn as nn
import torch.optim as optim
from models_mae import MaskedAutoencoderViT
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'util'))
from crfed_utils import ModelEncoder, mse_loss, model_to_vector

def train_model_encoder():
    """训练ModelEncoder"""
    print("开始训练ModelEncoder...")
    
    # 检查GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建小模型用于训练
    model = MaskedAutoencoderViT(
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=384,  # 使用更小的模型
        depth=6,
        num_heads=6,
        decoder_embed_dim=256,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4,
    ).to(device)
    
    # 获取模型参数
    theta = model_to_vector(model)
    input_dim = theta.numel()
    print(f"模型参数维度: {input_dim}")
    
    # 创建ModelEncoder（使用小架构）
    encoder = ModelEncoder(input_dim, bottleneck_dim=16, use_small_arch=True).to(device)
    
    # 优化器
    optimizer = optim.Adam(encoder.parameters(), lr=0.001)
    
    # 训练
    print("开始训练...")
    for epoch in range(10):
        optimizer.zero_grad()
        
        # 前向传播
        phi_t, theta_hat = encoder(theta)
        loss = mse_loss(theta, theta_hat)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
    
    print("ModelEncoder训练完成！")
    
    # 保存模型
    torch.save(encoder.state_dict(), 'model_encoder.pth')
    print("ModelEncoder已保存到 model_encoder.pth")
    
    return encoder

if __name__ == "__main__":
    train_model_encoder() 