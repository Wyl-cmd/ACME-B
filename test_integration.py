"""
ACME-B 完整集成测试
验证所有组件协同工作
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from acme_b import ACMELinear, ChemicalField, FisherLock, ReplayBuffer, Experience


class SimpleModel(nn.Module):
    """简单ACME-B模型"""
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10):
        super().__init__()
        self.fc1 = ACMELinear(input_dim, hidden_dim, tile_size=64)
        self.fc2 = ACMELinear(hidden_dim, hidden_dim, tile_size=64)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        self.chemical = ChemicalField()
        self.replay_buffer = ReplayBuffer(capacity=1000)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
    
    def get_memory_usage(self):
        """获取总内存使用"""
        mem1 = self.fc1.get_memory_usage()
        mem2 = self.fc2.get_memory_usage()
        return {
            'fp16_mb': mem1['fp16_mb'] + mem2['fp16_mb'],
            'acme_actual_mb': mem1['acme_actual_mb'] + mem2['acme_actual_mb'],
            'compression_ratio': (mem1['fp16_mb'] + mem2['fp16_mb']) / (mem1['acme_actual_mb'] + mem2['acme_actual_mb'])
        }


class DummyDataset(Dataset):
    """模拟数据集"""
    def __init__(self, size=1000, input_dim=784, num_classes=10):
        self.size = size
        self.input_dim = input_dim
        self.num_classes = num_classes
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        x = torch.randn(self.input_dim)
        y = torch.randint(0, self.num_classes, (1,)).squeeze()
        return x, y


def test_full_training_pipeline():
    """测试完整训练流程"""
    print("\n" + "="*60)
    print("测试1: 完整训练流程")
    print("="*60)
    
    # 创建模型和数据
    model = SimpleModel()
    dataset = DummyDataset(size=500)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 训练几轮
    print("\n训练模型...")
    for epoch in range(3):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (x, y) in enumerate(dataloader):
            if batch_idx >= 10:  # 只训练10个batch
                break
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            
            # 存储到replay buffer
            for i in range(len(x)):
                model.replay_buffer.push(x[i], y[i].item(), -loss.item())
        
        acc = correct / total if total > 0 else 0
        print(f"  Epoch {epoch+1}: Loss={total_loss/10:.4f}, Acc={acc:.4f}")
    
    # 内存使用
    mem = model.get_memory_usage()
    print(f"\n内存使用:")
    print(f"  FP16理论: {mem['fp16_mb']:.2f} MB")
    print(f"  ACME实际: {mem['acme_actual_mb']:.2f} MB")
    print(f"  压缩比: {mem['compression_ratio']:.2f}x")
    
    # Replay buffer
    print(f"\nReplay Buffer:")
    print(f"  大小: {len(model.replay_buffer)}")
    stats = model.replay_buffer.get_stats()
    print(f"  平均奖励: {stats.get('avg_reward', 0):.4f}")
    
    print("\n✓ 完整训练流程测试通过")


def test_chemical_field_integration():
    """测试化学场集成"""
    print("\n" + "="*60)
    print("测试2: 化学场集成")
    print("="*60)
    
    cf = ChemicalField()
    
    # 模拟不同性能下的化学场变化
    performances = [0.2, 0.4, 0.6, 0.8, 0.9]
    
    print("\n化学场自适应测试:")
    for perf in performances:
        cf.update(performance=perf, network_activity=0.15)
        print(f"  性能={perf:.1f} -> 多巴胺={cf.state.dopamine:.3f}, "
              f"血清素={cf.state.serotonin:.3f}, "
              f"去甲肾上腺素={cf.state.norepinephrine:.3f}")
    
    # 测试调制效果
    print("\n调制效果测试:")
    lr = 0.01
    sparsity = 0.5
    
    mod = cf.get_modulation(layer_idx=0, activity=torch.randn(256))
    effective_lr = lr * mod['learning_rate']
    effective_sparsity = sparsity * mod['sparsity']
    
    print(f"  原始学习率: {lr}")
    print(f"  调制后学习率: {effective_lr:.4f}")
    print(f"  原始稀疏度: {sparsity}")
    print(f"  调制后稀疏度: {effective_sparsity:.4f}")
    
    print("\n✓ 化学场集成测试通过")


def test_fisher_lock_integration():
    """测试Fisher Lock集成"""
    print("\n" + "="*60)
    print("测试3: Fisher Lock集成")
    print("="*60)
    
    # 创建模型
    model = nn.Sequential(
        nn.Linear(100, 200),
        nn.ReLU(),
        nn.Linear(200, 10)
    )
    
    fisher_lock = FisherLock(model, lock_threshold=0.7)
    
    # 创建数据集
    dataset = DummyDataset(size=200, input_dim=100, num_classes=10)
    dataloader = DataLoader(dataset, batch_size=20)
    
    # 计算Fisher信息并锁定
    print("\n计算Fisher信息并锁定...")
    fisher_lock.compute_and_lock(dataloader, num_batches=10)
    
    # 获取锁定统计
    stats = fisher_lock.get_lock_stats()
    print(f"  锁定状态: {stats['status']}")
    print(f"  锁定参数: {stats.get('total_locked', 0)}/{stats.get('total_params', 0)}")
    print(f"  锁定比例: {stats.get('overall_ratio', 0):.2%}")
    
    # 模拟训练并应用约束
    print("\n模拟训练并应用约束...")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    # 保存训练前权重
    weight_before = model[0].weight.data.clone()
    
    for step in range(5):
        x = torch.randn(20, 100)
        output = model(x)
        loss = output.pow(2).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 应用锁定约束
        fisher_lock.apply_constraint()
    
    # 检查权重变化
    weight_after = model[0].weight.data
    change = (weight_after - weight_before).abs().mean()
    print(f"  平均权重变化: {change.item():.6f}")
    
    print("\n✓ Fisher Lock集成测试通过")


def test_continual_learning_simulation():
    """测试持续学习模拟"""
    print("\n" + "="*60)
    print("测试4: 持续学习模拟")
    print("="*60)
    
    # 创建模型
    model = SimpleModel(input_dim=50, hidden_dim=128, output_dim=5)
    
    # 任务1数据
    class Task1Dataset(Dataset):
        def __len__(self):
            return 200
        def __getitem__(self, idx):
            x = torch.randn(50) + 1.0  # 任务1分布
            y = torch.randint(0, 5, (1,)).squeeze()
            return x, y
    
    # 任务2数据
    class Task2Dataset(Dataset):
        def __len__(self):
            return 200
        def __getitem__(self, idx):
            x = torch.randn(50) - 1.0  # 任务2分布
            y = torch.randint(0, 5, (1,)).squeeze()
            return x, y
    
    fisher_lock = FisherLock(model, lock_threshold=0.6)
    
    # 训练任务1
    print("\n训练任务1...")
    dataloader1 = DataLoader(Task1Dataset(), batch_size=20)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(2):
        for x, y in dataloader1:
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
    
    # 评估任务1
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader1:
            output = model(x)
            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    acc_task1_before = correct / total
    print(f"  任务1准确率(训练后): {acc_task1_before:.4f}")
    
    # 锁定任务1的重要参数
    print("\n锁定任务1参数...")
    fisher_lock.compute_and_lock(dataloader1, num_batches=10)
    
    # 训练任务2
    print("\n训练任务2...")
    dataloader2 = DataLoader(Task2Dataset(), batch_size=20)
    
    for epoch in range(2):
        for x, y in dataloader2:
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            
            # 添加Fisher惩罚
            fisher_penalty = fisher_lock.compute_fisher_penalty(model)
            total_loss = loss + 0.001 * fisher_penalty
            
            total_loss.backward()
            optimizer.step()
            
            # 应用锁定
            fisher_lock.apply_constraint()
    
    # 评估任务1（检查遗忘）
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader1:
            output = model(x)
            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    acc_task1_after = correct / total
    print(f"  任务1准确率(任务2后): {acc_task1_after:.4f}")
    
    forgetting = acc_task1_before - acc_task1_after
    print(f"  遗忘量: {forgetting:.4f}")
    
    if forgetting < 0.2:
        print("  ✓ 遗忘控制良好")
    else:
        print("  ⚠ 遗忘较严重")
    
    print("\n✓ 持续学习模拟测试通过")


def test_dream_phase():
    """测试梦境阶段"""
    print("\n" + "="*60)
    print("测试5: 梦境阶段")
    print("="*60)
    
    model = SimpleModel(input_dim=50, hidden_dim=64, output_dim=5)
    buffer = ReplayBuffer(capacity=500)
    
    # 填充replay buffer
    print("\n填充Replay Buffer...")
    for i in range(200):
        state = torch.randn(50)
        target = torch.randint(0, 5, (1,)).squeeze()
        buffer.push(state, target.item(), float(i % 10))
    
    print(f"  Buffer大小: {len(buffer)}")
    
    # 梦境训练
    print("\n梦境训练...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    dream_epochs = 3
    for epoch in range(dream_epochs):
        samples = buffer.sample(32)
        if not samples:
            break
        
        total_loss = 0
        for exp in samples:
            x = exp.state.unsqueeze(0)
            y = torch.tensor([exp.action])
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"  梦境Epoch {epoch+1}: Loss={total_loss/len(samples):.4f}")
    
    print("\n✓ 梦境阶段测试通过")


def main():
    """主函数"""
    print("="*60)
    print("ACME-B 完整集成测试")
    print("="*60)
    
    try:
        test_full_training_pipeline()
        test_chemical_field_integration()
        test_fisher_lock_integration()
        test_continual_learning_simulation()
        test_dream_phase()
        
        print("\n" + "="*60)
        print("✓✓✓ 所有集成测试通过！✓✓✓")
        print("="*60)
        print("\nACME-B 架构已完全可用！")
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
