"""
ACME-B 第三纪元: Fisher信息矩阵与权重锁定

防止灾难性遗忘的核心机制
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import copy


class FisherInformation:
    """
    Fisher信息矩阵估计器
    
    估计参数的重要性，用于决定哪些权重应该被锁定
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.fisher_dict: Dict[str, torch.Tensor] = {}
        self.param_count = sum(p.numel() for p in model.parameters())
        
    def compute_fisher(self, dataloader, num_batches: int = 100, device='cpu'):
        """
        计算Fisher信息矩阵 (对角近似)
        
        Fisher_{ii} = E[(dL/dw_i)^2]
        
        Args:
            dataloader: 数据加载器
            num_batches: 计算使用的批次数量
            device: 计算设备
        """
        self.model.eval()
        
        # 初始化Fisher矩阵
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.fisher_dict[name] = torch.zeros_like(param)
        
        count = 0
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            
            # 准备输入
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
            else:
                inputs = batch
            
            inputs = inputs.to(device)
            
            self.model.zero_grad()
            
            # 前向传播
            outputs = self.model(inputs)
            
            # 计算损失 (使用log likelihood)
            if outputs.dim() > 1:
                # 分类任务: 使用softmax交叉熵
                loss = -torch.log_softmax(outputs, dim=1).mean()
            else:
                # 回归任务: 使用MSE
                loss = outputs.pow(2).mean()
            
            # 反向传播
            loss.backward()
            
            # 累积梯度平方
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.fisher_dict[name] += param.grad.pow(2)
            
            count += 1
        
        # 平均
        for name in self.fisher_dict:
            self.fisher_dict[name] /= count
        
        print(f"Computed Fisher information from {count} batches")
    
    def get_importance(self, param_name: str) -> torch.Tensor:
        """获取参数的重要性"""
        if param_name in self.fisher_dict:
            return self.fisher_dict[param_name]
        else:
            return torch.zeros(1)
    
    def get_param_importance_score(self, param_name: str) -> float:
        """获取参数的整体重要性分数"""
        if param_name not in self.fisher_dict:
            return 0.0
        
        fisher = self.fisher_dict[param_name]
        # 使用Fisher的均值作为重要性
        return fisher.mean().item()


class FisherLock:
    """
    Fisher锁定机制
    
    基于Fisher信息锁定重要权重，防止灾难性遗忘
    """
    
    def __init__(
        self,
        model: nn.Module,
        lock_threshold: float = 0.7,
        consolidation_strength: float = 0.9,
        importance_threshold: float = 0.01  # 兼容旧API
    ):
        """
        Args:
            model: 要保护的模型
            lock_threshold: 锁定阈值 (Fisher分位数)
            consolidation_strength: 固化强度 (0-1)
            importance_threshold: 重要性阈值 (兼容参数)
        """
        self.model = model
        self.lock_threshold = lock_threshold
        self.consolidation_strength = consolidation_strength
        self.importance_threshold = importance_threshold
        
        self.fisher_info = FisherInformation(model)
        self.locked_params: Dict[str, torch.Tensor] = {}  # 锁定掩码
        self.original_values: Dict[str, torch.Tensor] = {}  # 原始值
        
        self.is_locked = False
    
    def update_fisher(self, model, dataloader, device='cpu', num_samples=100):
        """
        更新Fisher信息 (兼容旧API)
        
        Args:
            model: 模型
            dataloader: 数据加载器
            device: 设备
            num_samples: 样本数量
        """
        num_batches = num_samples // dataloader.batch_size if dataloader.batch_size else 10
        self.fisher_info.compute_fisher(dataloader, num_batches, device)
    
    def compute_fisher_penalty(self, model) -> torch.Tensor:
        """
        计算Fisher惩罚 (兼容旧API)
        
        Args:
            model: 模型
        
        Returns:
            惩罚值
        """
        if not self.is_locked or not self.locked_params:
            return torch.tensor(0.0)
        
        penalty = 0.0
        for name, param in model.named_parameters():
            if name in self.locked_params and name in self.original_values:
                fisher = self.fisher_info.get_importance(name)
                diff = (param - self.original_values[name]).pow(2)
                penalty += (fisher * diff).sum()
        
        return penalty
    
    def lock_important_parameters(self, model, lock_threshold=0.5):
        """
        锁定重要参数 (兼容旧API)
        
        Args:
            model: 模型
            lock_threshold: 锁定阈值
        
        Returns:
            锁定的参数数量
        """
        if not self.fisher_info.fisher_dict:
            return 0
        
        locked_count = 0
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            fisher = self.fisher_info.get_importance(name)
            
            if fisher.numel() > 0:
                threshold_val = torch.quantile(fisher.view(-1), lock_threshold)
                lock_mask = (fisher >= threshold_val).float()
                
                if lock_mask.sum() > 0:
                    self.locked_params[name] = lock_mask
                    self.original_values[name] = param.data.clone()
                    locked_count += int(lock_mask.sum().item())
        
        self.is_locked = True
        return locked_count
    
    def compute_and_lock(self, dataloader, num_batches: int = 100, device='cpu'):
        """
        计算Fisher信息并锁定重要权重
        
        Args:
            dataloader: 数据加载器
            num_batches: 计算批次
            device: 计算设备
        """
        print("Computing Fisher information...")
        self.fisher_info.compute_fisher(dataloader, num_batches, device)
        
        print("Locking important parameters...")
        self._apply_lock()
        
        self.is_locked = True
        
        # 统计
        total_locked = sum(mask.sum().item() for mask in self.locked_params.values())
        total_params = sum(mask.numel() for mask in self.locked_params.values())
        print(f"Locked {total_locked}/{total_params} parameters ({total_locked/total_params:.2%})")
    
    def _apply_lock(self):
        """应用锁定"""
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            fisher = self.fisher_info.get_importance(name)
            
            # 计算锁定阈值
            if fisher.numel() > 0:
                # 确保fisher是float类型
                fisher_float = fisher.float()
                threshold = torch.quantile(fisher_float.view(-1), self.lock_threshold)
                
                # 创建锁定掩码 (Fisher > threshold)
                lock_mask = (fisher >= threshold).float()
                
                self.locked_params[name] = lock_mask
                self.original_values[name] = param.data.clone()
            else:
                self.locked_params[name] = torch.zeros_like(param)
    
    def apply_constraint(self):
        """
        应用锁定约束
        
        在每次优化步骤后调用，确保被锁定的权重不偏离太远
        """
        if not self.is_locked:
            return
        
        for name, param in self.model.named_parameters():
            if name in self.locked_params and name in self.original_values:
                lock_mask = self.locked_params[name]
                original = self.original_values[name]
                
                # 混合: 锁定部分使用原始值，未锁定部分使用当前值
                param.data = lock_mask * (
                    self.consolidation_strength * original + 
                    (1 - self.consolidation_strength) * param.data
                ) + (1 - lock_mask) * param.data
    
    def get_lock_stats(self) -> Dict:
        """获取锁定统计"""
        if not self.is_locked:
            return {'status': 'not_locked'}
        
        stats = {
            'status': 'locked',
            'layers': []
        }
        
        for name, mask in self.locked_params.items():
            locked_count = mask.sum().item()
            total_count = mask.numel()
            
            stats['layers'].append({
                'name': name,
                'locked': locked_count,
                'total': total_count,
                'ratio': locked_count / total_count if total_count > 0 else 0
            })
        
        total_locked = sum(s['locked'] for s in stats['layers'])
        total_params = sum(s['total'] for s in stats['layers'])
        
        stats['total_locked'] = total_locked
        stats['total_params'] = total_params
        stats['overall_ratio'] = total_locked / total_params if total_params > 0 else 0
        
        return stats
    
    def unlock_all(self):
        """解锁所有参数"""
        self.locked_params.clear()
        self.original_values.clear()
        self.is_locked = False
        print("All parameters unlocked")


class ElasticWeightConsolidation:
    """
    弹性权重固化 (EWC)
    
    Kirkpatrick et al. 2017
    另一种防止遗忘的方法
    """
    
    def __init__(self, model: nn.Module, lambda_ewc: float = 1000.0):
        """
        Args:
            model: 模型
            lambda_ewc: EWC正则化强度
        """
        self.model = model
        self.lambda_ewc = lambda_ewc
        
        self.fisher_dict: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}
    
    def update_fisher(self, dataloader, num_batches: int = 100, device='cpu'):
        """更新Fisher信息"""
        fisher_info = FisherInformation(self.model)
        fisher_info.compute_fisher(dataloader, num_batches, device)
        self.fisher_dict = fisher_info.fisher_dict
        
        # 保存当前参数作为最优参数
        for name, param in self.model.named_parameters():
            self.optimal_params[name] = param.data.clone()
    
    def compute_ewc_loss(self) -> torch.Tensor:
        """
        计算EWC损失
        
        L_ewc = sum(F_i * (theta_i - theta*_i)^2)
        """
        if not self.fisher_dict:
            return torch.tensor(0.0)
        
        loss = 0.0
        for name, param in self.model.named_parameters():
            if name in self.fisher_dict and name in self.optimal_params:
                fisher = self.fisher_dict[name]
                optimal = self.optimal_params[name]
                
                loss += (fisher * (param - optimal).pow(2)).sum()
        
        return self.lambda_ewc * loss


# 测试代码
if __name__ == '__main__':
    print("=" * 60)
    print("ACME-B 第三纪元: Fisher锁定测试")
    print("=" * 60)
    
    # 创建简单模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(768, 3072)
            self.fc2 = nn.Linear(3072, 768)
            self.fc3 = nn.Linear(768, 10)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)
    
    model = SimpleModel()
    
    # 创建模拟数据
    class DummyDataset:
        def __init__(self, size=100):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            return torch.randn(768), torch.randint(0, 10, (1,)).squeeze()
    
    from torch.utils.data import DataLoader
    dataset = DummyDataset()
    dataloader = DataLoader(dataset, batch_size=16)
    
    # 测试Fisher信息计算
    print("\n1. Fisher信息计算:")
    fisher_info = FisherInformation(model)
    fisher_info.compute_fisher(dataloader, num_batches=10)
    
    for name, fisher in fisher_info.fisher_dict.items():
        print(f"  {name}: mean={fisher.mean():.6f}, max={fisher.max():.6f}")
    
    # 测试Fisher锁定
    print("\n2. Fisher锁定:")
    fisher_lock = FisherLock(model, lock_threshold=0.7)
    fisher_lock.compute_and_lock(dataloader, num_batches=10)
    
    stats = fisher_lock.get_lock_stats()
    print(f"  锁定比例: {stats['overall_ratio']:.2%}")
    print(f"  锁定参数: {stats['total_locked']}/{stats['total_params']}")
    
    # 测试EWC
    print("\n3. 弹性权重固化 (EWC):")
    ewc = ElasticWeightConsolidation(model, lambda_ewc=1000.0)
    ewc.update_fisher(dataloader, num_batches=10)
    
    ewc_loss = ewc.compute_ewc_loss()
    print(f"  EWC损失: {ewc_loss.item():.4f}")
    
    # 模拟训练并检查锁定效果
    print("\n4. 锁定效果测试:")
    
    # 保存锁定前的值
    fc1_weight_before = model.fc1.weight.data.clone()
    
    # 模拟优化步骤
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    for step in range(5):
        inputs = torch.randn(16, 768)
        outputs = model(inputs)
        loss = outputs.pow(2).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 应用锁定
        fisher_lock.apply_constraint()
    
    # 检查变化
    fc1_weight_after = model.fc1.weight.data
    change = (fc1_weight_after - fc1_weight_before).abs().mean()
    
    print(f"  平均权重变化: {change.item():.6f}")
    print(f"  (锁定后变化应小于未锁定)")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
