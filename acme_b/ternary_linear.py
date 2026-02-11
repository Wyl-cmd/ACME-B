"""
ACME-B 第一纪元: 三态推理核心
三态线性层实现 - Ternary Linear Layer

核心公式:
    Y = X · W_ternary
    W_ternary ∈ {-1, 0, +1}

存储优化:
    - FP16: 16 bits/weight
    - 三态: 2 bits/weight (1位符号 + 1位存在)
    - 压缩比: 8x
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
from typing import Optional, Tuple, Dict


class TernaryLinearFunction(Function):
    """
    三态线性函数 (PyTorch Autograd Function)
    
    前向传播: 使用三态权重计算
    反向传播: 使用STE (Straight-Through Estimator) 近似梯度
    """
    
    @staticmethod
    def forward(ctx, input, weight_base, weight_tag, mask):
        """
        前向传播
        
        Args:
            input: [batch, in_features]
            weight_base: 三态基础权重 [out_features, in_features], 取值 {-1, 0, +1}
            weight_tag: 标记权重 [out_features, in_features]
            mask: 瓦片掩码 [out_features, in_features], 取值 {0, 1}
        
        Returns:
            output: [batch, out_features]
        """
        # 保存用于反向传播
        ctx.save_for_backward(input, weight_base, weight_tag, mask)
        
        # 计算有效权重: W_effective = (W_base + W_tag) * mask
        weight_effective = (weight_base + weight_tag) * mask
        
        # 三态化 (确保输出严格三态)
        weight_effective = torch.clamp(weight_effective, -1, 1)
        weight_effective = torch.where(
            torch.abs(weight_effective) < 0.3,
            torch.zeros_like(weight_effective),
            torch.sign(weight_effective)
        )
        
        # 线性变换: Y = X @ W^T
        output = torch.matmul(input, weight_effective.t())
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播 - 使用STE (Straight-Through Estimator)
        
        由于三态函数不可导，我们使用STE:
        - 前向: 三态量化
        - 反向: 直接传递梯度 (identity)
        """
        input, weight_base, weight_tag, mask = ctx.saved_tensors
        
        # 对input的梯度 (正常计算)
        grad_input = torch.matmul(grad_output, (weight_base + weight_tag) * mask)
        
        # 对weight_tag的梯度 (STE: 直接传递)
        grad_weight_tag = torch.matmul(grad_output.t(), input) * mask
        
        # 对weight_base的梯度 (不更新，基础权重通过固化机制改变)
        grad_weight_base = None
        
        # 对mask的梯度 (稀疏性梯度)
        grad_mask = None  # mask通过拓扑演化改变，不通过梯度
        
        return grad_input, grad_weight_base, grad_weight_tag, grad_mask


class ACMELinear(nn.Module):
    """
    ACME-B 三态线性层
    
    权重三重表示:
        W_total = W_base + W_tag (由mask控制稀疏性)
    
    参数:
        in_features: 输入维度
        out_features: 输出维度
        tile_size: 瓦片大小 (默认64)
        bias: 是否使用偏置
        init_sparsity: 初始稀疏度
    
    示例:
        >>> layer = ACMELinear(768, 3072, tile_size=64)
        >>> x = torch.randn(2, 768)
        >>> y = layer(x)
        >>> print(y.shape)  # [2, 3072]
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        tile_size: int = 64,
        bias: bool = True,
        init_sparsity: float = 0.5,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.tile_size = tile_size
        self.init_sparsity = init_sparsity
        
        # 计算瓦片数量
        self.n_tiles_h = (out_features + tile_size - 1) // tile_size
        self.n_tiles_w = (in_features + tile_size - 1) // tile_size
        
        # === 第一纪元: 三态权重 ===
        
        # W_base: 三态基础权重 {-1, 0, +1}
        # 使用int8存储节省内存 (实际值: -1, 0, 1)
        self.register_buffer(
            'weight_base',
            torch.zeros(out_features, in_features, dtype=torch.int8)
        )
        
        # W_tag: 标记权重 (连续值，用于短期学习)
        # 使用FP32存储以确保梯度计算正确
        self.weight_tag = nn.Parameter(
            torch.zeros(out_features, in_features, dtype=torch.float32)
        )
        
        # Mask: 瓦片掩码 (0/1)
        # 每个瓦片一个掩码值
        self.register_buffer(
            'tile_mask',
            torch.ones(self.n_tiles_h, self.n_tiles_w, dtype=torch.float32)
        )
        
        # 初始化掩码 (随机稀疏)
        self._init_mask()
        
        # 初始化基础权重 (Xavier-like，但三态)
        self._init_weight_base()
        
        # 偏置
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # 统计信息
        self.register_buffer('tile_activation_count', 
                           torch.zeros(self.n_tiles_h, self.n_tiles_w))
        self.register_buffer('forward_count', torch.tensor(0))
        
    def _init_mask(self):
        """初始化瓦片掩码"""
        # 随机稀疏初始化
        prob = 1 - self.init_sparsity
        self.tile_mask.data = (torch.rand_like(self.tile_mask) < prob).float()
    
    def _init_weight_base(self):
        """初始化三态基础权重"""
        # Xavier-like初始化，但量化为三态
        limit = np.sqrt(6.0 / (self.in_features + self.out_features))
        
        # 随机初始化 [-1, 1]
        weight = torch.rand(self.out_features, self.in_features) * 2 - 1
        
        # 三态量化
        weight_ternary = self._ternarize(weight, threshold=limit)
        
        self.weight_base.data = weight_ternary.to(torch.int8)
    
    def _ternarize(self, x, threshold=0.3):
        """
        三态量化
        
        Args:
            x: 输入张量
            threshold: 量化阈值
        
        Returns:
            三态张量: -1, 0, +1
        """
        return torch.where(
            x > threshold,
            torch.ones_like(x),
            torch.where(
                x < -threshold,
                -torch.ones_like(x),
                torch.zeros_like(x)
            )
        ).to(torch.int8)
    
    def _expand_mask(self):
        """将瓦片掩码扩展到完整权重尺寸"""
        mask_full = torch.zeros(self.out_features, self.in_features, 
                               device=self.tile_mask.device)
        
        for i in range(self.n_tiles_h):
            for j in range(self.n_tiles_w):
                h_start = i * self.tile_size
                h_end = min((i + 1) * self.tile_size, self.out_features)
                w_start = j * self.tile_size
                w_end = min((j + 1) * self.tile_size, self.in_features)
                
                mask_full[h_start:h_end, w_start:w_end] = self.tile_mask[i, j]
        
        return mask_full
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [batch, in_features]
        
        Returns:
            output: [batch, out_features]
        """
        # 扩展掩码
        mask_full = self._expand_mask()
        
        # 转换weight_base为float用于计算
        weight_base_float = self.weight_base.float()
        weight_tag_float = self.weight_tag.float()
        
        # 三态线性变换
        output = TernaryLinearFunction.apply(
            x, weight_base_float, weight_tag_float, mask_full
        )
        
        # 添加偏置
        if self.bias is not None:
            output = output + self.bias
        
        # 更新统计
        self.forward_count += 1
        
        return output
    
    def consolidate_tags(self, consolidation_ratio=0.1):
        """
        将标记权重固化到基础权重 (梦境机制的一部分)
        
        Args:
            consolidation_ratio: 固化比例
        """
        with torch.no_grad():
            # 计算有效权重
            effective_weight = self.weight_base.float() + self.weight_tag.float()
            
            # 找出变化大的权重
            weight_change = torch.abs(self.weight_tag.float())
            
            # 选择Top-K变化最大的权重进行固化
            k = int(consolidation_ratio * weight_change.numel())
            if k > 0:
                topk_values, topk_indices = torch.topk(weight_change.view(-1), k)
                
                # 固化: W_base = ternarize(W_base + W_tag)
                new_base = self._ternarize(effective_weight)
                self.weight_base.data = new_base
                
                # 清空已固化的标记
                flat_mask = torch.zeros_like(weight_change.view(-1))
                flat_mask[topk_indices] = 1
                mask_2d = flat_mask.view_as(self.weight_tag)
                
                self.weight_tag.data *= (1 - mask_2d)
    
    def get_sparsity(self):
        """获取当前稀疏度"""
        active_tiles = self.tile_mask.sum().item()
        total_tiles = self.tile_mask.numel()
        return 1 - (active_tiles / total_tiles)
    
    def get_memory_usage(self):
        """
        获取内存使用量 (对比FP16)
        
        Returns:
            dict: 各种格式的内存使用量 (MB)
        """
        n_params = self.out_features * self.in_features
        
        # FP16
        fp16_bytes = n_params * 2
        
        # 三态 (2 bits = 0.25 bytes per param)
        ternary_bytes = n_params * 0.25
        
        # 实际存储 (int8 for base + fp16 for tag + float32 for mask)
        actual_bytes = (
            n_params * 1 +  # weight_base (int8)
            n_params * 2 +  # weight_tag (fp16)
            self.tile_mask.numel() * 4  # tile_mask (float32)
        )
        
        return {
            'fp16_mb': fp16_bytes / (1024 ** 2),
            'ternary_theoretical_mb': ternary_bytes / (1024 ** 2),
            'acme_actual_mb': actual_bytes / (1024 ** 2),
            'compression_ratio': fp16_bytes / actual_bytes,
        }
    
    def extra_repr(self):
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'tile_size={self.tile_size}, sparsity={self.get_sparsity():.2%}, '
                f'bias={self.bias is not None}')


# 测试代码
if __name__ == '__main__':
    print("=" * 60)
    print("ACME-B 第一纪元: 三态线性层测试")
    print("=" * 60)
    
    # 创建层
    layer = ACMELinear(768, 3072, tile_size=64, init_sparsity=0.3)
    print(f"\n层配置: {layer}")
    
    # 内存使用
    mem_usage = layer.get_memory_usage()
    print(f"\n内存使用:")
    print(f"  FP16理论: {mem_usage['fp16_mb']:.2f} MB")
    print(f"  三态理论: {mem_usage['ternary_theoretical_mb']:.2f} MB")
    print(f"  ACME实际: {mem_usage['acme_actual_mb']:.2f} MB")
    print(f"  压缩比: {mem_usage['compression_ratio']:.2f}x")
    
    # 前向传播测试
    x = torch.randn(2, 768)
    y = layer(x)
    print(f"\n前向传播:")
    print(f"  输入: {x.shape}")
    print(f"  输出: {y.shape}")
    print(f"  输出范围: [{y.min():.2f}, {y.max():.2f}]")
    
    # 统计信息
    print(f"\n统计信息:")
    print(f"  稀疏度: {layer.get_sparsity():.2%}")
    print(f"  活跃瓦片: {layer.tile_mask.sum().item()}/{layer.tile_mask.numel()}")
    print(f"  W_base唯一值: {torch.unique(layer.weight_base).tolist()}")
    
    # 固化测试
    print(f"\n固化测试:")
    print(f"  固化前 W_tag 范数: {layer.weight_tag.norm().item():.4f}")
    layer.consolidate_tags(consolidation_ratio=0.1)
    print(f"  固化后 W_tag 范数: {layer.weight_tag.norm().item():.4f}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
