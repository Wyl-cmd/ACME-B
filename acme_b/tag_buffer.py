"""
ACME-B 第二纪元: 标记系统 (Tagging System)
实现局部学习和短期记忆机制

核心组件:
- TagBuffer: 标记缓冲区
- ForwardForwardLayer: 前向-前向学习层
- LocalLearning: 局部学习协调器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import numpy as np


class TagBuffer:
    """
    标记缓冲区
    
    存储每层的短期记忆 (T_tag)
    作为W_base的动态补充
    """
    
    def __init__(
        self,
        layer_shape: Tuple[int, int],
        init_scale: float = 0.01,
        decay_rate: float = 0.99
    ):
        """
        Args:
            layer_shape: (out_features, in_features)
            init_scale: 初始缩放
            decay_rate: 标记衰减率
        """
        self.shape = layer_shape
        self.decay_rate = decay_rate
        
        # 标记权重 (短期记忆)
        self.T_tag = nn.Parameter(
            torch.randn(*layer_shape) * init_scale,
            requires_grad=True
        )
        
        # 标记历史 (用于分析)
        self.tag_history = []
        self.max_history = 100
        
        # 更新统计
        self.update_count = 0
        self.total_gradient_norm = 0.0
    
    def update(self, gradient: torch.Tensor, learning_rate: float = 0.01):
        """
        更新标记
        
        Args:
            gradient: 梯度张量
            learning_rate: 学习率
        """
        # 应用衰减
        self.T_tag.data *= self.decay_rate
        
        # 应用梯度
        with torch.no_grad():
            self.T_tag.data -= learning_rate * gradient
            
            # 限制范围 [-0.5, 0.5]
            self.T_tag.data = torch.clamp(self.T_tag.data, -0.5, 0.5)
        
        # 记录历史
        self.update_count += 1
        self.total_gradient_norm += gradient.norm().item()
        
        if len(self.tag_history) < self.max_history:
            self.tag_history.append(self.T_tag.data.clone())
    
    def get_effective_weight(self, weight_base: torch.Tensor) -> torch.Tensor:
        """
        获取有效权重
        
        W_effective = W_base + T_tag
        """
        return weight_base + self.T_tag
    
    def consolidate(self, consolidation_ratio: float = 0.1) -> torch.Tensor:
        """
        固化标记到基础权重
        
        Args:
            consolidation_ratio: 固化比例
        
        Returns:
            需要固化的权重增量
        """
        with torch.no_grad():
            # 选择变化最大的标记
            tag_magnitude = self.T_tag.abs()
            
            k = int(consolidation_ratio * self.T_tag.numel())
            threshold = torch.topk(tag_magnitude.view(-1), k).values[-1]
            
            # 创建固化掩码
            consolidate_mask = (tag_magnitude >= threshold).float()
            
            # 计算固化量
            delta = self.T_tag * consolidate_mask
            
            # 清空已固化的标记
            self.T_tag.data *= (1 - consolidate_mask)
            
            return delta
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'mean': self.T_tag.mean().item(),
            'std': self.T_tag.std().item(),
            'max': self.T_tag.max().item(),
            'min': self.T_tag.min().item(),
            'norm': self.T_tag.norm().item(),
            'update_count': self.update_count,
            'avg_gradient_norm': self.total_gradient_norm / max(1, self.update_count)
        }


class ForwardForwardLayer(nn.Module):
    """
    前向-前向学习层
    
    Hinton提出的局部学习算法
    每层独立学习，无需全局反向传播
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        positivity_threshold: float = 2.0,
        learning_rate: float = 0.01
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.positivity_threshold = positivity_threshold
        
        # 权重
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # 局部优化器
        self.local_lr = learning_rate
        
        # 统计
        self.positive_passes = 0
        self.negative_passes = 0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """标准前向传播"""
        return F.linear(x, self.weight, self.bias)
    
    def compute_goodness(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算"goodness" (激活强度)
        
        goodness = sum(activations^2)
        """
        activations = self.forward(x)
        goodness = activations.pow(2).sum(dim=1)
        return goodness
    
    def local_update(self, positive_data: torch.Tensor, negative_data: torch.Tensor):
        """
        局部更新 (Forward-Forward)
        
        Args:
            positive_data: 正样本 (真实数据)
            negative_data: 负样本 (预测错误或噪声)
        """
        # 计算goodness
        pos_goodness = self.compute_goodness(positive_data)
        neg_goodness = self.compute_goodness(negative_data)
        
        # FF损失: 希望正样本goodness高，负样本goodness低
        loss = torch.relu(self.positivity_threshold - pos_goodness).mean() + \
               torch.relu(neg_goodness - self.positivity_threshold).mean()
        
        # 局部梯度下降 (只更新当前层！)
        self.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            self.weight.data -= self.local_lr * self.weight.grad
            self.bias.data -= self.local_lr * self.bias.grad
        
        # 统计
        self.positive_passes += len(positive_data)
        self.negative_passes += len(negative_data)
        
        return loss.item()
    
    def generate_negative_data(self, positive_data: torch.Tensor, method: str = 'shuffle') -> torch.Tensor:
        """
        生成负样本
        
        Args:
            positive_data: 正样本
            method: 生成方法 ('shuffle', 'noise', 'mixup')
        """
        if method == 'shuffle':
            # 打乱特征顺序
            negative = positive_data.clone()
            for i in range(len(negative)):
                perm = torch.randperm(negative.shape[1])
                negative[i] = negative[i, perm]
            return negative
        
        elif method == 'noise':
            # 添加噪声
            noise = torch.randn_like(positive_data) * 0.5
            return positive_data + noise
        
        elif method == 'mixup':
            # MixUp
            alpha = 0.4
            lam = np.random.beta(alpha, alpha)
            batch_size = len(positive_data)
            index = torch.randperm(batch_size)
            mixed = lam * positive_data + (1 - lam) * positive_data[index]
            return mixed
        
        else:
            raise ValueError(f"Unknown method: {method}")


class LocalLearningCoordinator:
    """
    局部学习协调器
    
    协调多层的局部学习
    无需全局反向传播
    """
    
    def __init__(self, layers: List[ForwardForwardLayer]):
        self.layers = layers
        self.layer_losses = [[] for _ in layers]
    
    def train_step(self, data: torch.Tensor, negative_method: str = 'shuffle'):
        """
        训练一步
        
        Args:
            data: 输入数据
            negative_method: 负样本生成方法
        """
        # 逐层局部学习
        current_data = data
        
        for i, layer in enumerate(self.layers):
            # 生成负样本
            negative_data = layer.generate_negative_data(current_data, negative_method)
            
            # 局部更新
            loss = layer.local_update(current_data, negative_data)
            self.layer_losses[i].append(loss)
            
            # 前向传播到下一层
            with torch.no_grad():
                current_data = layer(current_data)
    
    def get_stats(self) -> Dict:
        """获取训练统计"""
        stats = {}
        for i, losses in enumerate(self.layer_losses):
            if losses:
                stats[f'layer_{i}'] = {
                    'recent_loss': losses[-1],
                    'avg_loss': np.mean(losses[-100:]),
                    'total_updates': len(losses)
                }
        return stats


class ACMELayerWithTag(nn.Module):
    """
    带标记的ACME层
    
    结合三态权重和标记系统
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        tile_size: int = 64,
        use_forward_forward: bool = False
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.use_forward_forward = use_forward_forward
        
        # 从第一纪元导入
        from .ternary_linear import ACMELinear
        self.linear = ACMELinear(in_features, out_features, tile_size=tile_size)
        
        # 标记缓冲区
        self.tag_buffer = TagBuffer((out_features, in_features))
        
        # 可选: Forward-Forward学习
        if use_forward_forward:
            self.ff_layer = ForwardForwardLayer(in_features, out_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 获取有效权重
        weight_base = self.linear.weight_base.float()
        weight_tag = self.tag_buffer.T_tag
        
        # 合并
        effective_weight = weight_base + weight_tag
        
        # 线性变换
        output = F.linear(x, effective_weight, self.linear.bias)
        
        return output
    
    def update_tags(self, gradient: torch.Tensor, learning_rate: float = 0.01):
        """更新标记"""
        self.tag_buffer.update(gradient, learning_rate)
    
    def consolidate_tags(self, ratio: float = 0.1):
        """固化标记到基础权重"""
        delta = self.tag_buffer.consolidate(ratio)
        
        # 更新基础权重
        with torch.no_grad():
            new_base = self.linear.weight_base.float() + delta
            self.linear.weight_base.data = self.linear._ternarize(new_base)


# 测试代码
if __name__ == '__main__':
    print("=" * 60)
    print("ACME-B 第二纪元: 标记系统测试")
    print("=" * 60)
    
    # 测试TagBuffer
    print("\n1. TagBuffer测试:")
    tag_buffer = TagBuffer((3072, 768), init_scale=0.01)
    
    # 模拟梯度更新
    for i in range(10):
        gradient = torch.randn(3072, 768) * 0.1
        tag_buffer.update(gradient, learning_rate=0.01)
    
    stats = tag_buffer.get_stats()
    print(f"  标记统计:")
    print(f"    mean={stats['mean']:.4f}, std={stats['std']:.4f}")
    print(f"    norm={stats['norm']:.4f}")
    print(f"    updates={stats['update_count']}")
    
    # 测试固化
    print(f"\n  固化测试:")
    print(f"    固化前范数: {tag_buffer.T_tag.norm().item():.4f}")
    delta = tag_buffer.consolidate(consolidation_ratio=0.1)
    print(f"    固化量范数: {delta.norm().item():.4f}")
    print(f"    固化后范数: {tag_buffer.T_tag.norm().item():.4f}")
    
    # 测试ForwardForwardLayer
    print("\n2. Forward-Forward层测试:")
    ff_layer = ForwardForwardLayer(768, 3072)
    
    # 生成数据
    positive_data = torch.randn(32, 768)
    
    # 训练几步
    for step in range(5):
        loss = ff_layer.local_update(
            positive_data,
            ff_layer.generate_negative_data(positive_data, 'shuffle')
        )
        print(f"  Step {step+1}: loss={loss:.4f}")
    
    # 测试完整层
    print("\n3. 完整ACME层测试:")
    acme_layer = ACMELayerWithTag(768, 3072, use_forward_forward=False)
    
    x = torch.randn(4, 768)
    y = acme_layer(x)
    
    print(f"  输入: {x.shape}")
    print(f"  输出: {y.shape}")
    print(f"  标记范数: {acme_layer.tag_buffer.T_tag.norm().item():.4f}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
