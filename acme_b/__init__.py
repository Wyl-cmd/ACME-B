"""
ACME-B: Autonomous Chemical-Morphological Evolution - Beta
基于Transformer的三态仿生架构

核心特性:
- 三态权重 (-1, 0, +1)
- 瓦片化稀疏计算
- 化学场调制
- 标记系统 (T_tag)
- 梦境固化机制
- 动态拓扑演化

五纪元架构:
- 第一纪元: 三态推理核心 (Ternary Inference Core) ✅
- 第二纪元: 标记系统 (Tagging System) ✅
- 第三纪元: 梦境固化 (Dream Consolidation) ✅
- 第四纪元: 拓扑演化 (Topology Evolution) 🔄
- 第五纪元: 化学觉醒 (Chemical Awakening) ✅

版本: 0.2.0-beta
"""

__version__ = "0.2.0-beta"
__author__ = "ACME-B Research Team"

# 第一纪元: 三态推理核心
from .ternary_linear import ACMELinear, TernaryLinearFunction

# 第二纪元: 标记系统 (简化版，集成在ACMELinear中)

# 第三纪元: 梦境固化
from .replay_buffer import ReplayBuffer, Experience
from .fisher_lock import FisherLock

# 第五纪元: 化学觉醒
from .chemical_field import ChemicalField

__all__ = [
    # 第一纪元
    'ACMELinear',
    'TernaryLinearFunction',
    
    # 第三纪元
    'ReplayBuffer',
    'Experience',
    'FisherLock',
    
    # 第五纪元
    'ChemicalField',
]
