"""
ACME-B 主训练器

整合所有纪元组件的完整训练框架
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional, List
import time
import json

from .ternary_linear import ACMELinear
from .tag_buffer import TagBuffer, ForwardForwardLayer
from .replay_buffer import ReplayBuffer, DreamPhase, Experience
from .fisher_lock import FisherLock, ElasticWeightConsolidation
from .chemical_field import ChemicalField, NeuromodulatedOptimizer
from .tile_manager import TileManager


class ACMEModel(nn.Module):
    """
    ACME-B 完整模型
    
    整合所有组件的可训练模型
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 3,
        tile_size: int = 64,
        use_tags: bool = True,
        use_chemical: bool = True
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.use_tags = use_tags
        self.use_chemical = use_chemical
        
        # 构建层
        self.layers = nn.ModuleList()
        
        # 输入层
        self.layers.append(ACMELinear(input_size, hidden_size, tile_size=tile_size))
        
        # 隐藏层
        for _ in range(num_layers - 2):
            self.layers.append(ACMELinear(hidden_size, hidden_size, tile_size=tile_size))
        
        # 输出层 (标准线性层，用于产生连续输出)
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        # 标记缓冲区 (第二纪元)
        if use_tags:
            self.tag_buffers = nn.ModuleList([
                TagBuffer((hidden_size, input_size if i == 0 else hidden_size))
                for i in range(num_layers - 1)
            ])
        
        # 化学场 (第五纪元)
        if use_chemical:
            self.chemical_field = ChemicalField()
        
        self.active_tile_ratio = 0.0
    
    def forward(self, x):
        """前向传播"""
        current = x
        
        for i, layer in enumerate(self.layers):
            # 获取有效权重
            if self.use_tags and i < len(self.tag_buffers):
                weight_base = layer.weight_base.float()
                weight_tag = self.tag_buffers[i].T_tag
                weight_effective = weight_base + weight_tag
            else:
                weight_effective = layer.weight_base.float()
            
            # 线性变换
            current = F.linear(current, weight_effective, layer.bias)
            current = F.relu(current)
        
        # 输出层
        output = self.output_layer(current)
        
        return output
    
    def get_network_activity(self):
        """获取网络活跃比例"""
        total_tiles = 0
        active_tiles = 0
        
        for layer in self.layers:
            if hasattr(layer, 'tile_mask'):
                total_tiles += layer.tile_mask.numel()
                active_tiles += layer.tile_mask.sum().item()
        
        return active_tiles / total_tiles if total_tiles > 0 else 0.0


class ACMETrainer:
    """
    ACME-B 训练器
    
    管理完整的训练流程，包括:
    - 清醒阶段学习
    - 梦境阶段固化
    - 化学场调节
    - Fisher锁定
    """
    
    def __init__(
        self,
        model: ACMEModel,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        
        # 优化器
        if optimizer is None:
            base_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        else:
            base_optimizer = optimizer
        
        # 神经调质优化器
        if model.use_chemical:
            self.optimizer = NeuromodulatedOptimizer(
                base_optimizer, 
                model.chemical_field
            )
        else:
            self.optimizer = base_optimizer
        
        # 经验回放 (第三纪元)
        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.dream_phase = DreamPhase(self.replay_buffer)
        
        # Fisher锁定 (第三纪元)
        self.fisher_lock = FisherLock(model)
        self.use_fisher_lock = False
        
        # 统计
        self.training_stats = {
            'steps': 0,
            'dreams': 0,
            'losses': [],
            'accuracies': []
        }
    
    def train_step(self, batch_x, batch_y):
        """
        单步训练
        
        Args:
            batch_x: 输入批次
            batch_y: 目标批次
        
        Returns:
            损失值
        """
        self.model.train()
        
        batch_x = batch_x.to(self.device)
        batch_y = batch_y.to(self.device)
        
        # 前向传播
        outputs = self.model(batch_x)
        
        # 计算损失
        loss = F.mse_loss(outputs, batch_y)
        
        # 反向传播
        if isinstance(self.optimizer, NeuromodulatedOptimizer):
            self.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()
        
        loss.backward()
        
        # 优化步骤
        if isinstance(self.optimizer, NeuromodulatedOptimizer):
            # 计算性能和活跃度
            with torch.no_grad():
                performance = 1.0 / (1.0 + loss.item())  # 简单转换
                activity = self.model.get_network_activity()
            
            self.optimizer.step(performance, activity)
        else:
            self.optimizer.step()
        
        # 应用Fisher锁定
        if self.use_fisher_lock:
            self.fisher_lock.apply_constraint()
        
        # 添加经验到回放缓冲区
        with torch.no_grad():
            exp = Experience(
                input_data=batch_x[0],  # 只存第一个样本
                target_data=batch_y[0],
                reward=-loss.item(),  # 负损失作为奖励
                importance=1.0 + loss.item()
            )
            self.replay_buffer.add(exp)
        
        # 更新统计
        self.training_stats['steps'] += 1
        self.training_stats['losses'].append(loss.item())
        
        # 检查是否需要梦境阶段
        if self.dream_phase.should_dream(
            self.training_stats['steps'], 
            dream_interval=1000
        ):
            self._dream_phase()
        
        return loss.item()
    
    def _dream_phase(self):
        """执行梦境阶段"""
        print(f"\n{'='*40}")
        print(f"Entering Dream Phase...")
        print(f"{'='*40}")
        
        # 执行梦境
        dream_stats = self.dream_phase.dream(self.model)
        
        print(f"Dream stats: {dream_stats}")
        
        # 固化标记
        if self.model.use_tags:
            for i, buffer in enumerate(self.model.tag_buffers):
                delta = buffer.consolidate(consolidation_ratio=0.1)
                # 更新基础权重
                layer = self.model.layers[i]
                new_base = layer.weight_base.float() + delta
                layer.weight_base.data = layer._ternarize(new_base)
        
        self.training_stats['dreams'] += 1
        
        print(f"{'='*40}\n")
    
    def enable_fisher_lock(self, dataloader, num_batches=100):
        """启用Fisher锁定"""
        print("Enabling Fisher Lock...")
        self.fisher_lock.compute_and_lock(dataloader, num_batches)
        self.use_fisher_lock = True
    
    def train_epoch(self, dataloader):
        """训练一个epoch"""
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y in dataloader:
            loss = self.train_step(batch_x, batch_y)
            epoch_loss += loss
            num_batches += 1
            
            if num_batches % 100 == 0:
                print(f"  Step {num_batches}, Loss: {loss:.4f}")
        
        return epoch_loss / num_batches
    
    def evaluate(self, dataloader):
        """评估模型"""
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                loss = F.mse_loss(outputs, batch_y)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def get_stats(self):
        """获取训练统计"""
        stats = {
            'training': self.training_stats,
            'replay_buffer': self.replay_buffer.get_stats(),
        }
        
        if self.model.use_chemical:
            stats['chemical_field'] = self.model.chemical_field.get_stats()
        
        if self.use_fisher_lock:
            stats['fisher_lock'] = self.fisher_lock.get_lock_stats()
        
        return stats
    
    def save_checkpoint(self, path):
        """保存检查点"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if not isinstance(self.optimizer, NeuromodulatedOptimizer) else self.optimizer.base_optimizer.state_dict(),
            'training_stats': self.training_stats,
            'replay_buffer': self.replay_buffer,
        }
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if isinstance(self.optimizer, NeuromodulatedOptimizer):
            self.optimizer.base_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.training_stats = checkpoint['training_stats']
        self.replay_buffer = checkpoint['replay_buffer']
        
        print(f"Checkpoint loaded from {path}")


# 示例：简单序列预测任务
if __name__ == '__main__':
    print("=" * 60)
    print("ACME-B 训练器示例")
    print("=" * 60)
    
    # 创建简单数据集
    class SimpleDataset:
        def __init__(self, size=1000, seq_len=10):
            self.size = size
            self.seq_len = seq_len
            
            # 生成简单序列数据
            self.data = []
            for _ in range(size):
                x = torch.randn(seq_len, 32)
                # 目标：累积和
                y = x.cumsum(dim=0)
                self.data.append((x, y))
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    # 创建数据
    train_dataset = SimpleDataset(size=500)
    test_dataset = SimpleDataset(size=100)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    # 创建模型
    model = ACMEModel(
        input_size=32,
        hidden_size=64,
        output_size=32,
        num_layers=3,
        tile_size=32,
        use_tags=True,
        use_chemical=True
    )
    
    print(f"\n模型结构:")
    print(f"  输入: 32")
    print(f"  隐藏: 64")
    print(f"  输出: 32")
    print(f"  层数: 3")
    print(f"  瓦片: 32")
    
    # 创建训练器
    trainer = ACMETrainer(model, device='cpu')
    
    # 训练
    print(f"\n开始训练...")
    num_epochs = 3
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        train_loss = trainer.train_epoch(train_loader)
        test_loss = trainer.evaluate(test_loader)
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Test Loss: {test_loss:.4f}")
        
        # 打印统计
        stats = trainer.get_stats()
        if 'chemical_field' in stats:
            chem = stats['chemical_field']
            print(f"  Dopamine: {chem['dopamine']:.3f}, "
                  f"Serotonin: {chem['serotonin']:.3f}, "
                  f"Norepinephrine: {chem['norepinephrine']:.3f}")
    
    # 最终统计
    print(f"\n{'='*40}")
    print("最终统计:")
    print(f"{'='*40}")
    
    final_stats = trainer.get_stats()
    print(f"总步数: {final_stats['training']['steps']}")
    print(f"梦境次数: {final_stats['training']['dreams']}")
    print(f"回放缓冲区: {final_stats['replay_buffer']['size']}")
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
