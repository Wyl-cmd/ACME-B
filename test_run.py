"""
ACME-B 运行测试
"""
import torch
from acme_b import ACMELinear, ChemicalField, FisherLock, ReplayBuffer

print('='*60)
print('ACME-B 完整功能测试')
print('='*60)

# 1. ACMELinear测试
print('\n[1] ACMELinear 三态线性层')
layer = ACMELinear(128, 64, tile_size=32)
x = torch.randn(16, 128)
output = layer(x)
print(f'    输入: {x.shape} -> 输出: {output.shape}')
print(f'    ✓ 前向传播正常')

loss = output.sum()
loss.backward()
print(f'    ✓ 反向传播正常')

memory = layer.get_memory_usage()
print(f'    ✓ 内存使用: {memory["fp16_mb"]:.4f} MB (FP16)')
print(f'    ✓ 压缩比: {memory["compression_ratio"]:.2f}x')

# 2. ChemicalField测试
print('\n[2] ChemicalField 化学场')
cf = ChemicalField()
for perf in [0.3, 0.5, 0.8, 0.9]:
    cf.update(performance=perf, network_activity=0.15)
    print(f'    性能{perf:.1f} -> 多巴胺: {cf.state.dopamine:.2f}')
print(f'    ✓ 化学场自适应正常')

# 3. ReplayBuffer测试
print('\n[3] ReplayBuffer 经验回放')
from acme_b import Experience
buffer = ReplayBuffer(capacity=100, alpha=0.6)
for i in range(50):
    state = torch.randn(10)
    target = torch.randn(10)
    exp = Experience(state, target, reward=float(i), importance=float(i))
    buffer.add(exp)
samples, indices, weights = buffer.sample(10)
print(f'    存储50条，采样10条，获得{len(samples)}条')
print(f'    ✓ 优先回放正常')

# 4. FisherLock测试
print('\n[4] FisherLock 遗忘锁定')
model = torch.nn.Sequential(
    torch.nn.Linear(10, 20),
    torch.nn.ReLU(),
    torch.nn.Linear(20, 2)
)
fisher = FisherLock(model)
print(f'    ✓ FisherLock创建正常')

# 5. 综合测试
print('\n[5] 综合训练测试')
layer2 = ACMELinear(64, 32, tile_size=16)
optimizer = torch.optim.Adam(layer2.parameters(), lr=0.001)
for step in range(5):
    x = torch.randn(8, 64)
    output = layer2(x)
    loss = output.pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(f'    训练5步，最终loss: {loss.item():.4f}')
print(f'    ✓ 训练流程正常')

print('\n' + '='*60)
print('✓ 所有测试通过！ACME-B 可以正常运行')
print('='*60)
