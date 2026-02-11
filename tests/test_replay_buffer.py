"""
ACME-B 单元测试 - Replay Buffer和Fisher Lock
"""

import torch
import torch.nn as nn
import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acme_b import ReplayBuffer, Experience, FisherLock


class TestReplayBuffer(unittest.TestCase):
    """测试Replay Buffer"""
    
    def setUp(self):
        """设置测试环境"""
        self.capacity = 100
        self.buffer = ReplayBuffer(capacity=self.capacity)
    
    def test_push_and_len(self):
        """测试添加经验和长度"""
        # 添加一些经验 (使用push方法)
        for i in range(50):
            state = torch.randn(10)
            action = i % 10
            reward = float(i)
            self.buffer.push(state, action, reward)
        
        self.assertEqual(len(self.buffer), 50)
    
    def test_add_and_len(self):
        """测试add方法和长度"""
        # 添加一些经验 (使用add方法)
        for i in range(50):
            exp = Experience(
                state=torch.randn(10),
                action=i % 10,
                reward=float(i)
            )
            self.buffer.add(exp)
        
        self.assertEqual(len(self.buffer), 50)
    
    def test_capacity_limit(self):
        """测试容量限制"""
        # 添加超过容量的经验
        for i in range(self.capacity + 50):
            self.buffer.push(torch.randn(10), i % 10, float(i))
        
        # 长度不应超过容量
        self.assertEqual(len(self.buffer), self.capacity)
    
    def test_sample(self):
        """测试采样"""
        # 添加经验
        for i in range(50):
            self.buffer.push(torch.randn(10), i % 10, float(i))
        
        # 采样
        batch_size = 10
        samples = self.buffer.sample(batch_size)
        
        self.assertEqual(len(samples), batch_size)
        
        # 检查每个样本
        for exp in samples:
            self.assertIsInstance(exp, Experience)
            self.assertEqual(exp.state.shape, (10,))
    
    def test_sample_empty(self):
        """测试空buffer采样"""
        samples = self.buffer.sample(10)
        self.assertEqual(len(samples), 0)
    
    def test_sample_more_than_available(self):
        """测试采样数量超过可用数量"""
        # 只添加5个经验
        for i in range(5):
            self.buffer.push(torch.randn(10), i, float(i))
        
        # 请求10个
        samples = self.buffer.sample(10)
        self.assertEqual(len(samples), 5)
    
    def test_sample_return_indices(self):
        """测试采样返回索引"""
        # 添加经验
        for i in range(50):
            self.buffer.push(torch.randn(10), i % 10, float(i))
        
        # 采样并获取索引
        samples, indices, weights = self.buffer.sample(10, return_indices=True)
        
        self.assertEqual(len(samples), 10)
        self.assertEqual(len(indices), 10)
        self.assertEqual(len(weights), 10)
    
    def test_update_priorities(self):
        """测试更新优先级"""
        # 添加经验
        for i in range(10):
            self.buffer.push(torch.randn(10), i, float(i))
        
        # 采样并获取indices
        samples, indices, weights = self.buffer.sample(5, return_indices=True)
        
        # 更新优先级
        new_priorities = [10.0] * len(indices)
        self.buffer.update_priorities(indices, new_priorities)
        
        # 检查优先级是否更新
        for idx in indices:
            self.assertEqual(self.buffer.priorities[idx], 10.0)
    
    def test_get_stats(self):
        """测试获取统计信息"""
        # 添加经验
        for i in range(50):
            self.buffer.push(torch.randn(10), i % 10, float(i), priority=float(i))
        
        stats = self.buffer.get_stats()
        
        self.assertIn('size', stats)
        self.assertIn('capacity', stats)
        self.assertEqual(stats['size'], 50)
        self.assertEqual(stats['capacity'], self.capacity)


class TestFisherLock(unittest.TestCase):
    """测试Fisher Lock"""
    
    def setUp(self):
        """设置测试环境"""
        torch.manual_seed(42)
        
        # 创建简单模型
        self.model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 2)
        )
        
        self.fisher_lock = FisherLock(self.model, lock_threshold=0.7)
    
    def test_initial_state(self):
        """测试初始状态"""
        # 初始时Fisher信息应该为空
        self.assertEqual(len(self.fisher_lock.fisher_info.fisher_dict), 0)
        
        # 惩罚应该为0
        penalty = self.fisher_lock.compute_fisher_penalty(self.model)
        self.assertEqual(penalty.item(), 0.0)
    
    def test_update_fisher(self):
        """测试更新Fisher信息"""
        # 创建简单数据集
        class SimpleDataset:
            def __init__(self, size=100):
                self.size = size
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                return torch.randn(10), torch.randint(0, 2, (1,)).squeeze()
        
        from torch.utils.data import DataLoader
        dataset = SimpleDataset()
        dataloader = DataLoader(dataset, batch_size=10)
        
        # 更新Fisher信息
        self.fisher_lock.update_fisher(self.model, dataloader, device='cpu', num_samples=50)
        
        # 检查Fisher信息是否被计算
        self.assertGreater(len(self.fisher_lock.fisher_info.fisher_dict), 0)
    
    def test_lock_important_parameters(self):
        """测试锁定重要参数"""
        # 创建简单数据集
        class SimpleDataset:
            def __len__(self):
                return 100
            
            def __getitem__(self, idx):
                return torch.randn(10), torch.randint(0, 2, (1,)).squeeze()
        
        from torch.utils.data import DataLoader
        dataset = SimpleDataset()
        dataloader = DataLoader(dataset, batch_size=10)
        
        # 更新Fisher信息
        self.fisher_lock.update_fisher(self.model, dataloader, 'cpu', num_samples=50)
        
        # 锁定参数
        locked_count = self.fisher_lock.lock_important_parameters(self.model, lock_threshold=0.5)
        
        # 检查是否有参数被锁定
        self.assertGreaterEqual(locked_count, 0)
        self.assertTrue(self.fisher_lock.is_locked)
    
    def test_get_lock_stats(self):
        """测试获取锁定统计"""
        # 未锁定时
        stats = self.fisher_lock.get_lock_stats()
        self.assertEqual(stats['status'], 'not_locked')


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def test_replay_with_model(self):
        """测试Replay Buffer与模型集成"""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 2)
        )
        
        buffer = ReplayBuffer(capacity=100)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        # 模拟训练并存储经验
        for i in range(20):
            x = torch.randn(5, 10)
            target = torch.randint(0, 2, (5,))
            
            # 前向传播
            output = model(x)
            loss = nn.functional.cross_entropy(output, target)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 存储经验
            for j in range(len(x)):
                buffer.push(x[j], target[j].item(), -loss.item())
        
        # 从重放缓冲区采样并重放
        samples = buffer.sample(10)
        self.assertEqual(len(samples), 10)
        
        # 使用采样进行训练
        for exp in samples:
            x = exp.state.unsqueeze(0)
            target = torch.tensor([exp.action])
            
            output = model(x)
            loss = nn.functional.cross_entropy(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def run_tests():
    """运行所有测试"""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    run_tests()
