"""
ACME-B 单元测试 - 三值线性层
"""

import torch
import torch.nn as nn
import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acme_b import ACMELinear, TernaryLinearFunction


class TestTernaryLinear(unittest.TestCase):
    """测试三值线性层"""
    
    def setUp(self):
        """设置测试环境"""
        torch.manual_seed(42)
        self.batch_size = 32
        self.in_features = 128
        self.out_features = 64
        self.tile_size = 32
    
    def test_forward_shape(self):
        """测试前向传播输出形状"""
        layer = ACMELinear(self.in_features, self.out_features, tile_size=self.tile_size)
        x = torch.randn(self.batch_size, self.in_features)
        
        output = layer(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.out_features))
    
    def test_ternary_weights(self):
        """测试权重是否被正确三值化"""
        layer = ACMELinear(self.in_features, self.out_features, tile_size=self.tile_size)
        
        # 获取三态化权重
        w_base = layer.weight_base.float()
        
        # 检查所有值都在 {-1, 0, +1} 中
        unique_values = torch.unique(w_base).tolist()
        expected_values = [-1., 0., 1.]
        
        for val in unique_values:
            self.assertIn(val, expected_values)
    
    def test_gradient_flow(self):
        """测试梯度是否能正确回流"""
        layer = ACMELinear(self.in_features, self.out_features, tile_size=self.tile_size)
        x = torch.randn(self.batch_size, self.in_features, requires_grad=True)
        
        output = layer(x)
        loss = output.sum()
        loss.backward()
        
        # 检查输入梯度是否存在
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.all(x.grad == 0))
        
        # 检查weight_tag梯度是否存在 (weight_base不更新)
        self.assertIsNotNone(layer.weight_tag.grad)
    
    def test_memory_usage(self):
        """测试内存使用报告"""
        layer = ACMELinear(self.in_features, self.out_features, tile_size=self.tile_size)
        
        memory = layer.get_memory_usage()
        
        # 内存使用应该返回字典
        self.assertIsInstance(memory, dict)
        self.assertIn('fp16_mb', memory)
        self.assertIn('compression_ratio', memory)
        
        # 内存使用应该大于0
        self.assertGreater(memory['fp16_mb'], 0)
    
    def test_sparsity(self):
        """测试稀疏度功能"""
        layer = ACMELinear(self.in_features, self.out_features, 
                          tile_size=self.tile_size, init_sparsity=0.5)
        
        sparsity = layer.get_sparsity()
        
        # 稀疏度应该在合理范围内
        self.assertGreaterEqual(sparsity, 0)
        self.assertLessEqual(sparsity, 1)
    
    def test_bias_option(self):
        """测试bias选项"""
        # 有bias
        layer_with_bias = ACMELinear(self.in_features, self.out_features, bias=True)
        self.assertIsNotNone(layer_with_bias.bias)
        
        # 无bias
        layer_without_bias = ACMELinear(self.in_features, self.out_features, bias=False)
        self.assertIsNone(layer_without_bias.bias)
    
    def test_consolidate_tags(self):
        """测试标记固化功能"""
        layer = ACMELinear(self.in_features, self.out_features, tile_size=self.tile_size)
        
        # 给weight_tag一些值
        layer.weight_tag.data = torch.randn_like(layer.weight_tag.data) * 0.1
        
        tag_norm_before = layer.weight_tag.norm().item()
        
        # 固化
        layer.consolidate_tags(consolidation_ratio=0.1)
        
        tag_norm_after = layer.weight_tag.norm().item()
        
        # 固化后tag应该减少
        self.assertLessEqual(tag_norm_after, tag_norm_before)


class TestACMEIntegration(unittest.TestCase):
    """测试ACME-B集成"""
    
    def test_simple_network(self):
        """测试简单网络的前向和反向传播"""
        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = ACMELinear(100, 64, tile_size=32)
                self.fc2 = ACMELinear(64, 10, tile_size=32)
            
            def forward(self, x):
                x = self.fc1(x)
                x = torch.relu(x)
                x = self.fc2(x)
                return x
        
        model = SimpleNet()
        x = torch.randn(32, 100)
        target = torch.randint(0, 10, (32,))
        
        # 前向传播
        output = model(x)
        self.assertEqual(output.shape, (32, 10))
        
        # 计算损失并反向传播
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        loss.backward()
        
        # 检查梯度
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
    
    def test_training_step(self):
        """测试单步训练"""
        model = ACMELinear(50, 20, tile_size=16)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        x = torch.randn(16, 50)
        target = torch.randn(16, 20)
        
        # 训练前获取权重
        w_tag_before = model.weight_tag.data.clone()
        
        # 训练步骤
        optimizer.zero_grad()
        output = model(x)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        optimizer.step()
        
        # 检查weight_tag是否更新 (weight_base不更新)
        w_tag_after = model.weight_tag.data
        self.assertFalse(torch.allclose(w_tag_before, w_tag_after))


def run_tests():
    """运行所有测试"""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    run_tests()
