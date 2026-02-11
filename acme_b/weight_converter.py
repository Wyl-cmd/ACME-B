"""
ACME-B 权重转换工具
将FP16/BF32权重转换为三态权重

支持:
- 直接转换 (阈值截断)
- 量化感知训练 (QAT)
- 渐进式转换
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict


class TernaryQuantizer:
    """
    三态量化器
    
    将连续权重转换为 {-1, 0, +1}
    """
    
    def __init__(self, threshold: float = 0.3, method: str = 'threshold'):
        """
        Args:
            threshold: 量化阈值
            method: 量化方法 ('threshold', 'scale', 'adaptive')
        """
        self.threshold = threshold
        self.method = method
    
    def quantize(self, weight: torch.Tensor) -> torch.Tensor:
        """
        量化权重到三态
        
        Args:
            weight: 输入权重 [任意形状]
        
        Returns:
            三态权重 {-1, 0, +1}
        """
        if self.method == 'threshold':
            return self._threshold_quantize(weight)
        elif self.method == 'scale':
            return self._scale_quantize(weight)
        elif self.method == 'adaptive':
            return self._adaptive_quantize(weight)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _threshold_quantize(self, weight: torch.Tensor) -> torch.Tensor:
        """阈值量化"""
        return torch.where(
            weight > self.threshold,
            torch.ones_like(weight),
            torch.where(
                weight < -self.threshold,
                -torch.ones_like(weight),
                torch.zeros_like(weight)
            )
        )
    
    def _scale_quantize(self, weight: torch.Tensor) -> torch.Tensor:
        """缩放后量化"""
        # 计算缩放因子
        scale = weight.abs().mean() * 2
        
        # 缩放并量化
        scaled = weight / scale
        return torch.where(
            scaled > self.threshold,
            torch.ones_like(weight),
            torch.where(
                scaled < -self.threshold,
                -torch.ones_like(weight),
                torch.zeros_like(weight)
            )
        )
    
    def _adaptive_quantize(self, weight: torch.Tensor) -> torch.Tensor:
        """自适应阈值量化"""
        # 根据权重的标准差自适应调整阈值
        std = weight.std()
        adaptive_threshold = self.threshold * std
        
        return torch.where(
            weight > adaptive_threshold,
            torch.ones_like(weight),
            torch.where(
                weight < -adaptive_threshold,
                -torch.ones_like(weight),
                torch.zeros_like(weight)
            )
        )
    
    def compute_error(self, original: torch.Tensor, quantized: torch.Tensor) -> float:
        """计算量化误差"""
        return (original - quantized).pow(2).mean().sqrt().item()


class WeightConverter:
    """
    权重转换器
    
    将PyTorch模型的权重转换为ACME-B格式
    """
    
    def __init__(self, quantizer: Optional[TernaryQuantizer] = None):
        self.quantizer = quantizer or TernaryQuantizer()
    
    def convert_layer(self, linear_layer: nn.Linear) -> Dict[str, torch.Tensor]:
        """
        转换单个线性层
        
        Args:
            linear_layer: PyTorch Linear层
        
        Returns:
            转换后的权重字典
        """
        weight = linear_layer.weight.data
        
        # 量化为基础权重
        weight_base = self.quantizer.quantize(weight)
        
        # 计算残差作为标记权重
        weight_tag = weight - weight_base
        
        # 限制tag范围
        weight_tag = torch.clamp(weight_tag, -0.5, 0.5)
        
        result = {
            'weight_base': weight_base.to(torch.int8),
            'weight_tag': weight_tag.to(torch.float16),
        }
        
        if linear_layer.bias is not None:
            result['bias'] = linear_layer.bias.data.clone()
        
        return result
    
    def convert_model(self, model: nn.Module, target_layers: Optional[list] = None) -> Dict[str, Dict]:
        """
        转换整个模型
        
        Args:
            model: PyTorch模型
            target_layers: 目标层名称列表，None则转换所有Linear层
        
        Returns:
            转换后的权重字典 {层名: {weight_base, weight_tag, bias}}
        """
        converted = {}
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if target_layers is None or name in target_layers:
                    converted[name] = self.convert_layer(module)
                    print(f"Converted: {name} -> shape: {module.weight.shape}")
        
        return converted
    
    def progressive_convert(self, model: nn.Module, steps: int = 10) -> nn.Module:
        """
        渐进式转换 (用于量化感知训练)
        
        逐步增加量化比例，让模型适应
        
        Args:
            model: 原始模型
            steps: 渐进步数
        
        Returns:
            转换后的模型
        """
        for step in range(1, steps + 1):
            ratio = step / steps
            
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    # 混合权重
                    original = module.weight.data
                    quantized = self.quantizer.quantize(original)
                    
                    # 渐进混合
                    mixed = original * (1 - ratio) + quantized * ratio
                    module.weight.data = mixed
            
            print(f"Progressive conversion: {ratio*100:.0f}%")
        
        return model
    
    def analyze_conversion(self, model: nn.Module) -> Dict:
        """
        分析转换效果
        
        Returns:
            分析统计信息
        """
        stats = {
            'layers': [],
            'total_params': 0,
            'total_error': 0,
        }
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.data
                quantized = self.quantizer.quantize(weight)
                
                layer_stats = {
                    'name': name,
                    'shape': list(weight.shape),
                    'params': weight.numel(),
                    'error': self.quantizer.compute_error(weight, quantized),
                    'sparsity': (quantized == 0).float().mean().item(),
                }
                
                stats['layers'].append(layer_stats)
                stats['total_params'] += layer_stats['params']
                stats['total_error'] += layer_stats['error']
        
        stats['avg_error'] = stats['total_error'] / len(stats['layers'])
        
        return stats


class ModelImporter:
    """
    模型导入器
    
    从HuggingFace等导入预训练模型并转换
    """
    
    def __init__(self, converter: Optional[WeightConverter] = None):
        self.converter = converter or WeightConverter()
    
    def from_huggingface(self, model_name: str, **kwargs):
        """
        从HuggingFace导入模型
        
        Args:
            model_name: 模型名称，如 'gpt2', 'bert-base-uncased'
        
        Returns:
            转换后的权重字典
        """
        try:
            from transformers import AutoModel
            
            print(f"Loading model from HuggingFace: {model_name}")
            model = AutoModel.from_pretrained(model_name, **kwargs)
            
            print("Converting to ACME-B format...")
            converted = self.converter.convert_model(model)
            
            return converted
        
        except ImportError:
            print("Please install transformers: pip install transformers")
            raise
    
    def from_pytorch(self, checkpoint_path: str):
        """
        从PyTorch checkpoint导入
        
        Args:
            checkpoint_path: 模型路径
        
        Returns:
            转换后的权重字典
        """
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 提取模型状态
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        print("Converting to ACME-B format...")
        
        # 转换每个权重
        converted = {}
        for name, param in state_dict.items():
            if 'weight' in name and param.ndim == 2:
                # 假设是线性层权重
                weight_base = self.converter.quantizer.quantize(param)
                weight_tag = param - weight_base
                
                converted[name] = {
                    'weight_base': weight_base.to(torch.int8),
                    'weight_tag': weight_tag.to(torch.float16),
                }
        
        return converted


# 测试代码
if __name__ == '__main__':
    print("=" * 60)
    print("ACME-B 权重转换工具测试")
    print("=" * 60)
    
    # 创建测试权重
    test_weight = torch.randn(768, 3072)
    print(f"\n测试权重: {test_weight.shape}")
    print(f"  原始范围: [{test_weight.min():.3f}, {test_weight.max():.3f}]")
    
    # 测试不同量化方法
    methods = ['threshold', 'scale', 'adaptive']
    
    for method in methods:
        print(f"\n{method.upper()} 量化:")
        quantizer = TernaryQuantizer(threshold=0.3, method=method)
        quantized = quantizer.quantize(test_weight)
        
        unique_values = torch.unique(quantized).tolist()
        error = quantizer.compute_error(test_weight, quantized)
        sparsity = (quantized == 0).float().mean().item()
        
        print(f"  唯一值: {unique_values}")
        print(f"  量化误差: {error:.4f}")
        print(f"  稀疏度: {sparsity:.2%}")
    
    # 测试线性层转换
    print("\n" + "-" * 60)
    print("Linear层转换测试:")
    
    linear = nn.Linear(768, 3072)
    converter = WeightConverter()
    converted = converter.convert_layer(linear)
    
    print(f"  W_base: {converted['weight_base'].shape}, dtype: {converted['weight_base'].dtype}")
    print(f"  W_tag: {converted['weight_tag'].shape}, dtype: {converted['weight_tag'].dtype}")
    
    # 验证转换
    reconstructed = converted['weight_base'].float() + converted['weight_tag'].float()
    original = linear.weight.data
    conversion_error = (original - reconstructed).pow(2).mean().sqrt()
    
    print(f"  重构误差: {conversion_error:.4f}")
    
    # 测试模型分析
    print("\n" + "-" * 60)
    print("模型分析测试:")
    
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(768, 3072)
            self.fc2 = nn.Linear(3072, 768)
        
        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))
    
    model = TestModel()
    stats = converter.analyze_conversion(model)
    
    print(f"  总参数量: {stats['total_params']:,}")
    print(f"  平均量化误差: {stats['avg_error']:.4f}")
    print(f"\n  各层详情:")
    for layer in stats['layers']:
        print(f"    {layer['name']}: {layer['shape']}, "
              f"error={layer['error']:.4f}, sparsity={layer['sparsity']:.2%}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
