"""
ACME-B 字符预测任务示例
展示如何使用ACME-B架构进行简单的字符级语言模型训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acme_b import (
    ACMELinear, TagBuffer, ChemicalField, 
    ReplayBuffer, DreamPhase, FisherLock
)


class CharDataset(Dataset):
    """字符数据集"""
    def __init__(self, text, seq_length):
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        self.seq_length = seq_length
        self.data = text
        
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        seq = self.data[idx:idx+self.seq_length]
        target = self.data[idx+1:idx+self.seq_length+1]
        
        x = torch.tensor([self.char_to_idx[ch] for ch in seq], dtype=torch.long)
        y = torch.tensor([self.char_to_idx[ch] for ch in target], dtype=torch.long)
        return x, y


class ACMELanguageModel(nn.Module):
    """ACME-B语言模型"""
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, 
                 num_layers=2, tile_size=64, use_tags=True, use_chemical=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.use_tags = use_tags
        self.use_chemical = use_chemical
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # ACME-B层
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = embed_dim if i == 0 else hidden_dim
            out_dim = hidden_dim
            
            layer = ACMELinear(
                in_features=in_dim,
                out_features=out_dim,
                tile_size=tile_size,
                use_tags=use_tags,
                tag_ratio=0.1,
                use_chemical=use_chemical
            )
            self.layers.append(layer)
        
        # 输出层
        self.output = nn.Linear(hidden_dim, vocab_size)
        
        # 化学场（全局共享）
        if use_chemical:
            self.chemical_field = ChemicalField(
                num_layers=num_layers,
                layer_dim=hidden_dim,
                modulation_strength=0.3
            )
        else:
            self.chemical_field = None
        
        # Replay Buffer用于梦境阶段
        self.replay_buffer = ReplayBuffer(capacity=10000)
        
    def forward(self, x, return_goodness=False):
        # 嵌入
        x = self.embedding(x)  # [batch, seq, embed]
        batch_size, seq_len, _ = x.shape
        
        # 展平处理
        x = x.view(batch_size * seq_len, -1)
        
        # 化学场状态
        if self.use_chemical and self.training:
            self.chemical_field.update_baseline(x)
        
        # 通过ACME-B层
        all_goodness = []
        for i, layer in enumerate(self.layers):
            if self.use_chemical:
                # 获取化学调制
                chemical_mod = self.chemical_field.get_modulation(i, x)
                x, goodness = layer(x, chemical_mod=chemical_mod)
            else:
                x, goodness = layer(x)
            
            x = F.relu(x)
            all_goodness.append(goodness)
        
        # 输出
        logits = self.output(x)
        logits = logits.view(batch_size, seq_len, self.vocab_size)
        
        if return_goodness:
            return logits, all_goodness
        return logits
    
    def get_memory_usage(self):
        """获取内存使用情况"""
        total_memory = 0
        for layer in self.layers:
            if isinstance(layer, ACMELinear):
                total_memory += layer.get_memory_usage()
        return total_memory
    
    def dream_phase(self, batch_size=32):
        """梦境阶段：重放历史经验"""
        if len(self.replay_buffer) < batch_size:
            return 0.0
        
        # 采样历史经验
        experiences = self.replay_buffer.sample(batch_size)
        
        # 这里可以添加梦境学习的逻辑
        # 例如：使用Forward-Forward算法进行离线学习
        
        return 0.0


def train_epoch(model, dataloader, optimizer, device, fisher_lock=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    
    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        
        # 前向传播
        logits = model(x)
        
        # 计算损失
        loss = F.cross_entropy(logits.view(-1, model.vocab_size), y.view(-1))
        
        # 添加Fisher惩罚（防止灾难性遗忘）
        if fisher_lock is not None:
            fisher_penalty = fisher_lock.compute_fisher_penalty(model)
            loss = loss + 0.01 * fisher_penalty
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        pred = logits.argmax(dim=-1)
        total_correct += (pred == y).sum().item()
        total_tokens += y.numel()
        
        # 存储到replay buffer
        if batch_idx % 10 == 0:
            model.replay_buffer.push(x.cpu(), y.cpu(), loss.item())
        
        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_tokens
    return avg_loss, accuracy


def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, model.vocab_size), y.view(-1))
            
            total_loss += loss.item()
            pred = logits.argmax(dim=-1)
            total_correct += (pred == y).sum().item()
            total_tokens += y.numel()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_tokens
    return avg_loss, accuracy


def generate_text(model, dataset, start_text="Hello", length=200, temperature=1.0, device='cpu'):
    """生成文本"""
    model.eval()
    
    # 编码起始文本
    input_seq = [dataset.char_to_idx[ch] for ch in start_text]
    input_tensor = torch.tensor([input_seq], dtype=torch.long).to(device)
    
    generated = start_text
    
    with torch.no_grad():
        # 预热
        hidden = None
        
        for _ in range(length):
            # 前向传播
            logits = model(input_tensor)
            
            # 获取最后一个字符的预测
            next_char_logits = logits[0, -1, :] / temperature
            probs = F.softmax(next_char_logits, dim=-1)
            
            # 采样
            next_char_idx = torch.multinomial(probs, num_samples=1).item()
            next_char = dataset.idx_to_char[next_char_idx]
            
            generated += next_char
            
            # 更新输入
            input_seq = input_seq[1:] + [next_char_idx]
            input_tensor = torch.tensor([input_seq], dtype=torch.long).to(device)
    
    return generated


def main():
    """主函数"""
    # 配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 示例文本（可以用更大的语料库替换）
    sample_text = """
    The quick brown fox jumps over the lazy dog.
    Machine learning is a subset of artificial intelligence.
    Neural networks are inspired by biological neurons.
    Deep learning has revolutionized computer vision and natural language processing.
    The future of AI is both exciting and uncertain.
    We must develop AI responsibly and ethically.
    """ * 100  # 重复以增加数据量
    
    # 超参数
    seq_length = 50
    batch_size = 16
    embed_dim = 64
    hidden_dim = 128
    num_layers = 2
    num_epochs = 10
    learning_rate = 0.001
    
    # 创建数据集
    dataset = CharDataset(sample_text, seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Vocab size: {dataset.vocab_size}")
    print(f"Dataset size: {len(dataset)}")
    
    # 创建模型
    model = ACMELanguageModel(
        vocab_size=dataset.vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        tile_size=64,
        use_tags=True,
        use_chemical=True
    ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Memory usage: {model.get_memory_usage() / 1024 / 1024:.2f} MB")
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Fisher Lock（可选，用于防止灾难性遗忘）
    fisher_lock = FisherLock(model, importance_threshold=0.01)
    
    # 训练循环
    print("\n" + "="*50)
    print("开始训练")
    print("="*50)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # 训练
        train_loss, train_acc = train_epoch(model, dataloader, optimizer, device, fisher_lock)
        
        # 评估
        val_loss, val_acc = evaluate(model, dataloader, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # 更新Fisher信息
        if epoch % 3 == 0:
            fisher_lock.update_fisher(model, dataloader, device, num_samples=100)
        
        # 梦境阶段
        if epoch % 2 == 0:
            print("进入梦境阶段...")
            model.dream_phase(batch_size=16)
        
        # 生成示例文本
        if epoch % 2 == 0:
            generated = generate_text(
                model, dataset, 
                start_text="The future", 
                length=100, 
                temperature=0.8,
                device=device
            )
            print(f"\n生成文本: {generated}\n")
    
    # 最终生成
    print("\n" + "="*50)
    print("最终生成示例")
    print("="*50)
    
    for start in ["The", "Machine", "AI", "Neural"]:
        generated = generate_text(
            model, dataset, 
            start_text=start, 
            length=150, 
            temperature=0.8,
            device=device
        )
        print(f"\n'{start}' -> {generated}")
    
    print("\n训练完成!")
    print(f"最终内存使用: {model.get_memory_usage() / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
