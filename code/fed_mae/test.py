import torch
from models_mae import MaskedAutoencoderViT
from torch.nn import LayerNorm
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'util'))
from crfed_utils import (
    model_to_vector, ModelEncoder, mse_loss, 
    indicator_function, optimal_sigma, sample_weight,
    concat_meta_indicator_batch, MetaProjector, get_default_tau_lambda
)

# 检查GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

model = MaskedAutoencoderViT(
    img_size=224,
    patch_size=16,
    in_chans=3,
    embed_dim=768,
    depth=12,
    num_heads=12,
    decoder_embed_dim=512,
    decoder_depth=8,
    decoder_num_heads=16,
    mlp_ratio=4,
    # norm_layer=LayerNorm(1e-6)
)

# 将模型移到GPU
model = model.to(device)


def test_crfed_utils():
    """测试CRFed工具函数"""
    print("=" * 50)
    print("开始测试CRFed工具...")
    
    # 1. 测试模型参数向量化
    print("\n1. 测试模型参数向量化:")
    theta = model_to_vector(model)
    print(f"模型参数向量维度: {theta.shape}")
    print(f"参数总数: {theta.numel()}")
    
    # 2. 测试ModelEncoder
    print("\n2. 测试ModelEncoder:")
    input_dim = theta.numel()
    # 使用小架构避免内存问题
    encoder = ModelEncoder(input_dim, bottleneck_dim=16, use_small_arch=True).to(device)
    phi_t, theta_hat = encoder(theta)
    print(f"输入维度: {input_dim}")
    print(f"Meta-model维度: {phi_t.shape}")
    print(f"重构维度: {theta_hat.shape}")
    
    # 3. 测试MSE loss
    print("\n3. 测试MSE loss:")
    loss = mse_loss(theta, theta_hat)
    print(f"重构损失: {loss.item():.6f}")
    
    # 4. 测试Indicator Function
    print("\n4. 测试Indicator Function:")
    tau, lambd = get_default_tau_lambda()
    print(f"tau={tau}, lambda={lambd}")
    
    # 模拟一些样本loss
    sample_losses = [0.1, 0.5, 1.0, 2.0, 5.0]
    print("样本loss:", sample_losses)
    
    for l_i in sample_losses:
        sigma_star = optimal_sigma(l_i, tau, lambd)
        I_lambda = indicator_function(l_i, sigma_star, lambd, tau)
        weight = sample_weight(l_i, tau, lambd)
        print(f"loss={l_i:.1f}: sigma*={sigma_star:.4f}, I_lambda={I_lambda:.4f}, weight={weight:.4f}")
    
    # 5. 测试MetaProjector
    print("\n5. 测试MetaProjector:")
    projector = MetaProjector(input_dim=16, proj_dim=64).to(device)
    P_phi_t = projector(phi_t)
    print(f"投影前维度: {phi_t.shape}")
    print(f"投影后维度: {P_phi_t.shape}")
    
    # 6. 测试拼接功能
    print("\n6. 测试拼接功能:")
    I_lambda_batch = torch.tensor([indicator_function(l_i, optimal_sigma(l_i, tau, lambd), lambd, tau) 
                                  for l_i in sample_losses], device=device)
    zi_batch = concat_meta_indicator_batch(P_phi_t, I_lambda_batch)
    print(f"I_lambda_batch维度: {I_lambda_batch.shape}")
    print(f"zi_batch维度: {zi_batch.shape}")
    print(f"拼接后的zi包含: meta-model特征({P_phi_t.shape[0]}维) + indicator(1维) = {zi_batch.shape[1]}维")
    
    # 7. 测试forward_with_fallback
    print("\n7. 测试forward_with_fallback:")
    try:
        result = projector.forward_with_fallback(phi_t)
        print(f"forward_with_fallback成功，结果维度: {result.shape}")
    except Exception as e:
        print(f"forward_with_fallback异常: {e}")
    
    print("\n" + "=" * 50)
    print("CRFed工具测试完成！")


if __name__ == "__main__":
    
    '''
    checkpoint = torch.load("D:/Dev/DiatomFL/data/checkpoints/mae_pretrain_vit_base.pth")
    state_dict = checkpoint["model"]
    # 只保留以 "blocks"、"patch_embed"、"norm"、"pos_embed"、"cls_token" 等开头的参数
    encoder_keys = [k for k in state_dict.keys() if not k.startswith("decoder")]
    encoder_state_dict = {k: v for k, v in state_dict.items() if k in encoder_keys}
    model.load_state_dict(encoder_state_dict, strict=False)

    # print(model)  # 打印模型结构

    print("已加载参数名：")
    for k in encoder_state_dict.keys():
        print(k)

    print("模型所有参数名：")
    for k in model.state_dict().keys():
        print(k)

    # print("patch_embed.proj.weight 的部分值：")
    # print(model.patch_embed.proj.weight[:2])  # 只看前两行，避免输出太多

    print("参数总数：", sum(p.numel() for p in model.parameters()))
    '''
    
    # 测试CRFed工具
    test_crfed_utils()








