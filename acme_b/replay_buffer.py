"""
ACME-B 第三纪元: 梦境固化 - Replay Buffer

实现经验重放缓冲区，用于离线学习和记忆巩固
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from collections import deque
import time


class Experience:
    """单个经验样本"""
    
    def __init__(
        self,
        state: torch.Tensor,
        action: Optional[Union[torch.Tensor, int]] = None,
        reward: float = 0.0,
        next_state: Optional[torch.Tensor] = None,
        done: bool = False,
        importance: float = 1.0
    ):
        """
        Args:
            state: 当前状态
            action: 动作 (可选)
            reward: 奖励
            next_state: 下一状态 (可选)
            done: 是否结束
            importance: 重要性
        """
        self.state = state.detach().cpu() if isinstance(state, torch.Tensor) else state
        self.action = action
        self.reward = reward
        self.next_state = next_state.detach().cpu() if isinstance(next_state, torch.Tensor) else next_state
        self.done = done
        self.importance = importance
        self.timestamp = time.time()
        self.access_count = 0
    
    def to(self, device):
        """移动到指定设备"""
        if isinstance(self.state, torch.Tensor):
            self.state = self.state.to(device)
        if isinstance(self.next_state, torch.Tensor):
            self.next_state = self.next_state.to(device)
        return self


class ReplayBuffer:
    """
    经验重放缓冲区
    
    存储和采样训练经验，支持优先级采样
    """
    
    def __init__(
        self,
        capacity: int = 10000,
        alpha: float = 0.6,  # 优先级指数
        beta: float = 0.4,   # 重要性采样指数
        beta_increment: float = 0.001
    ):
        """
        Args:
            capacity: 缓冲区容量
            alpha: 优先级程度 (0=均匀, 1=完全优先级)
            beta: 重要性采样纠正程度
            beta_increment: beta增长速率
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.beta_max = 1.0
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        
        # 统计
        self.total_experiences = 0
    
    def push(self, state, action=None, reward=0.0, next_state=None, done=False, priority=1.0):
        """
        添加经验到缓冲区 (兼容旧API)
        
        Args:
            state: 当前状态
            action: 动作 (可选)
            reward: 奖励
            next_state: 下一状态 (可选)
            done: 是否结束
            priority: 优先级
        """
        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            importance=priority
        )
        self.add(experience)
    
    def add(self, experience: Experience):
        """
        添加经验到缓冲区
        
        Args:
            experience: Experience对象
        """
        # 新经验使用最大优先级
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        self.total_experiences += 1
    
    def sample(self, batch_size: int, return_indices=False) -> Union[Tuple[List[Experience], np.ndarray, np.ndarray], List[Experience]]:
        """
        优先级采样
        
        Args:
            batch_size: 采样数量
            return_indices: 是否返回索引
        
        Returns:
            如果return_indices=True: (experiences, indices, importance_weights)
            否则: experiences
        """
        if len(self.buffer) == 0:
            if return_indices:
                return [], np.array([]), np.array([])
            return []
        
        # 如果请求的batch_size大于缓冲区大小，调整batch_size
        actual_batch_size = min(batch_size, len(self.buffer))
        
        # 计算采样概率
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # 采样
        indices = np.random.choice(len(self.buffer), actual_batch_size, p=probabilities, replace=False)
        experiences = [self.buffer[idx] for idx in indices]
        
        # 计算重要性权重
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # 归一化
        
        # 增加beta
        self.beta = min(self.beta_max, self.beta + self.beta_increment)
        
        # 更新访问计数
        for idx in indices:
            self.buffer[idx].access_count += 1
        
        if return_indices:
            return experiences, indices, weights
        return experiences
    
    def sample_indices(self, batch_size: int) -> np.ndarray:
        """
        仅采样索引
        
        Args:
            batch_size: 采样数量
        
        Returns:
            索引数组
        """
        if len(self.buffer) == 0:
            return np.array([])
        
        actual_batch_size = min(batch_size, len(self.buffer))
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        return np.random.choice(len(self.buffer), actual_batch_size, p=probabilities, replace=False)
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """更新经验优先级"""
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < len(self.buffer):
                self.priorities[idx] = priority
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        if not self.buffer:
            return {'size': 0}
        
        rewards = [exp.reward for exp in self.buffer]
        importances = [exp.importance for exp in self.buffer]
        access_counts = [exp.access_count for exp in self.buffer]
        
        return {
            'size': len(self.buffer),
            'capacity': self.capacity,
            'total_added': self.total_experiences,
            'avg_reward': np.mean(rewards) if rewards else 0,
            'avg_importance': np.mean(importances) if importances else 0,
            'avg_access_count': np.mean(access_counts) if access_counts else 0,
            'max_priority': self.priorities[:len(self.buffer)].max() if len(self.buffer) > 0 else 0,
            'min_priority': self.priorities[:len(self.buffer)].min() if len(self.buffer) > 0 else 0,
        }
    
    def __len__(self):
        return len(self.buffer)


class DreamPhase:
    """
    梦境阶段管理器
    
    控制睡眠/梦境周期的经验重放
    """
    
    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        dream_epochs: int = 5,
        dream_batch_size: int = 32
    ):
        self.replay_buffer = replay_buffer
        self.dream_epochs = dream_epochs
        self.dream_batch_size = dream_batch_size
        
        self.dream_count = 0
        self.total_dream_samples = 0
    
    def dream(self, model, optimizer=None):
        """
        执行梦境阶段
        
        Args:
            model: 要训练的模型
            optimizer: 优化器 (可选)
        
        Returns:
            梦境统计
        """
        if len(self.replay_buffer) < self.dream_batch_size:
            return {'status': 'not_enough_data'}
        
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(self.dream_epochs):
            # 采样经验
            experiences, indices, weights = self.replay_buffer.sample(self.dream_batch_size, return_indices=True)
            
            if not experiences:
                break
            
            # 准备数据
            inputs = torch.stack([exp.state for exp in experiences])
            
            # 前向传播
            outputs = model(inputs)
            
            # 计算损失 (自监督或MSE)
            loss = outputs.pow(2).mean()
            
            # 反向传播
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            self.total_dream_samples += len(experiences)
        
        self.dream_count += 1
        
        return {
            'status': 'success',
            'dream_count': self.dream_count,
            'avg_loss': total_loss / num_batches if num_batches > 0 else 0,
            'total_samples': self.total_dream_samples,
        }
    
    def should_dream(self, wake_steps: int, dream_interval: int = 1000) -> bool:
        """判断是否应该进入梦境阶段"""
        return wake_steps > 0 and wake_steps % dream_interval == 0


# 测试代码
if __name__ == '__main__':
    print("=" * 60)
    print("ACME-B 第三纪元: Replay Buffer测试")
    print("=" * 60)
    
    # 创建缓冲区
    buffer = ReplayBuffer(capacity=1000, alpha=0.6, beta=0.4)
    
    # 添加经验 (使用push方法)
    print("\n1. 添加经验 (push方法):")
    for i in range(100):
        buffer.push(
            state=torch.randn(768),
            action=i % 10,
            reward=np.random.randn(),
            priority=np.random.uniform(0.5, 1.5)
        )
    
    print(f"  缓冲区大小: {len(buffer)}")
    
    # 添加经验 (使用add方法)
    print("\n2. 添加经验 (add方法):")
    for i in range(50):
        exp = Experience(
            state=torch.randn(768),
            action=i % 10,
            reward=np.random.randn(),
            importance=np.random.uniform(0.5, 1.5)
        )
        buffer.add(exp)
    
    print(f"  缓冲区大小: {len(buffer)}")
    
    # 采样测试
    print("\n3. 优先级采样:")
    experiences, indices, weights = buffer.sample(batch_size=10, return_indices=True)
    print(f"  采样数量: {len(experiences)}")
    print(f"  重要性权重范围: [{weights.min():.3f}, {weights.max():.3f}]")
    
    # 统计
    print("\n4. 缓冲区统计:")
    stats = buffer.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
