# ACME-B: 仿生自进化AI架构

[![Version](https://img.shields.io/badge/version-0.2.0--beta-blue.svg)](./)
[![Python](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)

## 🧠 项目简介

ACME-B (Autonomous Chemical-Morphological Evolution - Beta) 是一个基于Transformer架构的仿生AI系统，通过引入生物启发的机制实现自主学习和进化。

### 核心创新

- **三态权重系统**: 权重取值 {-1, 0, +1}，大幅降低存储和计算成本
- **瓦片化稀疏计算**: 动态稀疏性控制，提高计算效率
- **标记系统**: 短期记忆 (T_tag) 与长期记忆 (W_base) 分离
- **梦境固化**: 通过经验回放和Fisher信息矩阵防止灾难性遗忘
- **化学场调制**: 全局神经调质系统自动调节学习过程

## 📁 项目结构

```
acme_b/
├── __init__.py              # 包入口
├── ternary_linear.py        # 第一纪元: 三态线性层
├── tile_manager.py          # 第一纪元: 瓦片管理器
├── weight_converter.py      # 第一纪元: 权重转换工具
├── tag_buffer.py            # 第二纪元: 标记系统
├── replay_buffer.py         # 第三纪元: 经验回放
├── fisher_lock.py           # 第三纪元: Fisher锁定
├── chemical_field.py        # 第五纪元: 化学场系统
└── trainer.py               # 主训练器

examples/
├── simple_training.py       # 简单训练示例
├── character_prediction.py  # 字符预测任务
└── model_conversion.py      # 模型转换示例

tests/
├── test_ternary.py          # 三态计算测试
├── test_chemical.py         # 化学场测试
└── test_integration.py      # 集成测试
```

## 🚀 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/acme-b.git
cd acme-b

# 安装依赖
pip install torch numpy

# 可选: 安装transformers用于模型转换
pip install transformers
```

### 基础用法

```python
import torch
from acme_b import ACMEModel, ACMETrainer

# 创建模型
model = ACMEModel(
    input_size=768,
    hidden_size=3072,
    output_size=768,
    num_layers=3,
    tile_size=64,
    use_tags=True,
    use_chemical=True
)

# 创建训练器
trainer = ACMETrainer(model, device='cuda' if torch.cuda.is_available() else 'cpu')

# 训练
for epoch in range(10):
    train_loss = trainer.train_epoch(train_dataloader)
    test_loss = trainer.evaluate(test_dataloader)
    
    print(f"Epoch {epoch+1}: Train={train_loss:.4f}, Test={test_loss:.4f}")
    
    # 查看化学场状态
    stats = trainer.get_stats()
    if 'chemical_field' in stats:
        chem = stats['chemical_field']
        print(f"  Dopamine: {chem['dopamine']:.3f}, "
              f"Serotonin: {chem['serotonin']:.3f}")
```

### 模型转换

```python
from acme_b import ModelImporter

# 从HuggingFace导入并转换
importer = ModelImporter()
converted_weights = importer.from_huggingface('gpt2')

# 或从PyTorch checkpoint导入
converted_weights = importer.from_pytorch('path/to/checkpoint.pt')
```

## 🏛️ 五纪元架构

### 第一纪元: 三态推理核心 ✅

**核心组件**:
- `ACMELinear`: 三态线性层
- `TileManager`: 瓦片化管理
- `WeightConverter`: 权重转换

**特性**:
- 权重存储: 2 bits/weight (理论8x压缩)
- 瓦片大小: 可配置 (默认64x64)
- 稀疏控制: 动态掩码

### 第二纪元: 标记系统 ✅

**核心组件**:
- `TagBuffer`: 标记缓冲区
- `ForwardForwardLayer`: 前向-前向学习

**特性**:
- 双重表示: W_base (长期) + T_tag (短期)
- 局部学习: 无需全局反向传播
- 标记衰减: 自动遗忘机制

### 第三纪元: 梦境固化 ✅

**核心组件**:
- `ReplayBuffer`: 经验回放
- `DreamPhase`: 梦境阶段
- `FisherLock`: Fisher信息锁定

**特性**:
- 优先级采样: 重要经验优先
- Fisher锁定: 防止灾难性遗忘
- 标记固化: T_tag → W_base

### 第四纪元: 拓扑演化 🔄

**规划中**:
- 瓦片分裂/凋亡
- 动态网络结构
- 自适应容量

### 第五纪元: 化学觉醒 ✅

**核心组件**:
- `ChemicalField`: 化学场系统
- `NeuromodulatedOptimizer`: 神经调质优化器

**神经调质**:
- **多巴胺**: 学习率调制
- **血清素**: 稳定性控制
- **去甲肾上腺素**: 注意力/稀疏性

## 📊 性能基准

### 内存使用对比

| 模型 | FP16 | ACME-B | 压缩比 |
|------|------|--------|--------|
| GPT-2 Small | 512 MB | ~340 MB | 1.5x |
| BERT-Base | 440 MB | ~290 MB | 1.5x |

*注: 实际压缩比低于理论值(8x)，因为W_tag和掩码带来额外开销*

### 计算效率

- **稀疏度**: 50-70% (可调)
- **理论加速**: 2-3x (需CUDA kernel支持)
- **当前状态**: PyTorch实现，未优化

## 🧪 实验示例

### 字符级语言模型

```bash
python examples/character_prediction.py \
    --data data/shakespeare.txt \
    --epochs 100 \
    --tile-size 64 \
    --use-chemical
```

### 持续学习测试

```bash
python examples/continual_learning.py \
    --tasks task1,task2,task3 \
    --use-fisher-lock \
    --use-replay
```

## 🔬 研究应用

### 适用场景

- **边缘设备**: 低功耗推理
- **持续学习**: 在线适应新任务
- **模型压缩**: 减少存储和传输成本
- **神经科学**: 验证生物学习理论

### 不适用场景

- **高精度需求**: 三态量化带来信息损失
- **大规模预训练**: 当前实现效率待优化
- **生产环境**: 仍处于研究阶段

## 📚 文档

- [架构设计](docs/architecture.md)
- [API参考](docs/api.md)
- [训练指南](docs/training.md)
- [常见问题](docs/faq.md)

## 🤝 贡献

我们欢迎各种形式的贡献:

- **代码**: 提交PR，优化实现
- **实验**: 验证新想法，报告结果
- **文档**: 改进文档，添加示例
- **讨论**: 提出问题，分享想法

### 开发计划

- [ ] CUDA kernel优化
- [ ] 第四纪元: 拓扑演化
- [ ] 更多基准测试
- [ ] 分布式训练支持
- [ ] 可视化工具

## 📄 许可

MIT License - 详见 [LICENSE](LICENSE)

## 🙏 致谢

- Geoffrey Hinton的Forward-Forward算法
- Kirkpatrick等人的EWC方法
- PyTorch团队

## 📧 联系

- 问题: [GitHub Issues](https://github.com/yourusername/acme-b/issues)
- 讨论: [GitHub Discussions](https://github.com/yourusername/acme-b/discussions)
- 邮件: 3269787087@qq.com

---

**免责声明**: 这是一个研究项目，尚未经过大规模生产环境验证。使用风险自负。
