"""
ACME-B 第五纪元: 化学觉醒 - 化学场系统

全局神经调质系统，自动调节网络状态
"""

import torch
import numpy as np
from typing import Dict, Optional, List
from dataclasses import dataclass


@dataclass
class ChemicalState:
    """化学场状态"""
    dopamine: float = 0.5      # 奖励信号 [0, 1]
    serotonin: float = 0.5     # 稳定性 [0, 1]
    norepinephrine: float = 0.3  # 注意力/唤醒 [0, 1]
    
    # 扩展调质
    acetylcholine: float = 0.5  # 学习可塑性
    gaba: float = 0.5           # 抑制性调节
    glutamate: float = 0.5      # 兴奋性调节


class ChemicalField:
    """
    化学场系统
    
    全局神经调质，自动调节:
    - 学习率 (多巴胺)
    - 稳定性 (血清素)
    - 注意力/稀疏性 (去甲肾上腺素)
    """
    
    def __init__(
        self,
        adaptation_rate: float = 0.01,
        target_activity: float = 0.1,
        homeostasis_range: tuple = (0.1, 0.9)
    ):
        """
        Args:
            adaptation_rate: 自适应速率
            target_activity: 目标活跃比例
            homeostasis_range: 稳态范围
        """
        self.state = ChemicalState()
        self.adaptation_rate = adaptation_rate
        self.target_activity = target_activity
        self.homeostasis_range = homeostasis_range
        
        # 历史记录
        self.history: List[ChemicalState] = []
        self.max_history = 1000
        
        # 性能历史
        self.performance_history: List[float] = []
        
    def update(self, performance: float, network_activity: float):
        """
        根据性能和网络活动更新化学场
        
        Args:
            performance: 当前性能 (0-1)
            network_activity: 网络活跃比例 (0-1)
        """
        self.performance_history.append(performance)
        
        # 多巴胺: 跟随性能
        if performance > 0.7:
            self.state.dopamine = min(1.0, self.state.dopamine + 0.05)
        elif performance < 0.3:
            self.state.dopamine = max(0.1, self.state.dopamine - 0.1)
        else:
            # 缓慢衰减
            self.state.dopamine *= 0.95
        
        # 血清素: 维持稳定性
        if len(self.performance_history) >= 10:
            recent_variance = np.var(self.performance_history[-10:])
            if recent_variance > 0.1:  # 不稳定
                self.state.serotonin = min(1.0, self.state.serotonin + 0.1)
            else:
                self.state.serotonin = max(0.2, self.state.serotonin - 0.02)
        
        # 去甲肾上腺素: 调节稀疏性
        if network_activity > self.target_activity:
            # 太活跃，降低唤醒
            self.state.norepinephrine = max(0.1, self.state.norepinephrine - 0.05)
        else:
            # 不够活跃，提高唤醒
            self.state.norepinephrine = min(0.9, self.state.norepinephrine + 0.03)
        
        # 乙酰胆碱: 学习可塑性 (与多巴胺相关)
        self.state.acetylcholine = 0.5 + 0.5 * self.state.dopamine
        
        # GABA: 抑制性 (与血清素相关)
        self.state.gaba = self.state.serotonin
        
        # 谷氨酸: 兴奋性 (与去甲肾上腺素相关)
        self.state.glutamate = self.state.norepinephrine
        
        # 记录历史
        self._record_history()
    
    def _record_history(self):
        """记录化学场历史"""
        self.history.append(ChemicalState(
            dopamine=self.state.dopamine,
            serotonin=self.state.serotonin,
            norepinephrine=self.state.norepinephrine,
            acetylcholine=self.state.acetylcholine,
            gaba=self.state.gaba,
            glutamate=self.state.glutamate
        ))
        
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def get_learning_rate_factor(self) -> float:
        """获取学习率调制因子"""
        # 多巴胺促进学习，血清素稳定学习
        return self.state.dopamine * (1 - self.state.serotonin * 0.5)
    
    def get_attention_boost(self) -> float:
        """获取注意力调制因子"""
        return self.state.norepinephrine
    
    def get_stability_factor(self) -> float:
        """获取稳定性因子"""
        return self.state.serotonin
    
    def get_plasticity_factor(self) -> float:
        """获取可塑性因子"""
        return self.state.acetylcholine
    
    def get_consolidation_threshold(self) -> float:
        """获取固化阈值"""
        # 血清素高时更容易固化
        return 0.5 + 0.4 * self.state.serotonin
    
    def get_modulation(self, layer_idx: int, activity: torch.Tensor) -> Dict:
        """
        获取指定层的调制参数
        
        Args:
            layer_idx: 层索引
            activity: 层激活值
        
        Returns:
            调制参数字典
        """
        # 学习率调制 (基于多巴胺和血清素)
        learning_rate = self.get_learning_rate_factor()
        
        # 稀疏性调制 (基于去甲肾上腺素)
        sparsity = 0.5 + 0.3 * (1 - self.state.norepinephrine)
        
        # 稳定性调制 (基于血清素)
        stability = self.state.serotonin
        
        # 注意力调制 (基于去甲肾上腺素)
        attention = self.state.norepinephrine
        
        # 可塑性调制 (基于乙酰胆碱)
        plasticity = self.state.acetylcholine
        
        return {
            'learning_rate': learning_rate,
            'sparsity': sparsity,
            'stability': stability,
            'attention': attention,
            'plasticity': plasticity,
            'dopamine': self.state.dopamine,
            'serotonin': self.state.serotonin,
            'norepinephrine': self.state.norepinephrine,
        }
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'dopamine': self.state.dopamine,
            'serotonin': self.state.serotonin,
            'norepinephrine': self.state.norepinephrine,
            'acetylcholine': self.state.acetylcholine,
            'gaba': self.state.gaba,
            'glutamate': self.state.glutamate,
            'learning_factor': self.get_learning_rate_factor(),
            'attention_boost': self.get_attention_boost(),
            'history_length': len(self.history),
        }
    
    def reset(self):
        """重置化学场"""
        self.state = ChemicalState()
        self.history.clear()
        self.performance_history.clear()


class NeuromodulatedOptimizer:
    """
    神经调质优化器包装器
    
    使用化学场自动调节优化器参数
    """
    
    def __init__(self, base_optimizer, chemical_field: ChemicalField):
        self.base_optimizer = base_optimizer
        self.chemical_field = chemical_field
        
        self.base_lr = base_optimizer.param_groups[0]['lr']
    
    def step(self, performance: float, network_activity: float):
        """优化步骤"""
        # 更新化学场
        self.chemical_field.update(performance, network_activity)
        
        # 调制学习率
        lr_factor = self.chemical_field.get_learning_rate_factor()
        new_lr = self.base_lr * lr_factor
        
        for param_group in self.base_optimizer.param_groups:
            param_group['lr'] = new_lr
        
        # 执行优化
        self.base_optimizer.step()
    
    def zero_grad(self):
        """清零梯度"""
        self.base_optimizer.zero_grad()


# 测试代码
if __name__ == '__main__':
    print("=" * 60)
    print("ACME-B 第五纪元: 化学场系统测试")
    print("=" * 60)
    
    # 创建化学场
    chem = ChemicalField()
    
    print("\n1. 化学场更新测试:")
    
    # 模拟不同性能场景
    scenarios = [
        (0.9, 0.15, "高性能, 正常活跃"),
        (0.2, 0.25, "低性能, 过度活跃"),
        (0.5, 0.05, "中等性能, 活跃不足"),
    ]
    
    for perf, activity, desc in scenarios:
        chem.reset()
        
        # 多次更新
        for _ in range(10):
            chem.update(perf, activity)
        
        stats = chem.get_stats()
        print(f"\n  {desc}:")
        print(f"    多巴胺: {stats['dopamine']:.3f}")
        print(f"    血清素: {stats['serotonin']:.3f}")
        print(f"    去甲肾上腺素: {stats['norepinephrine']:.3f}")
        print(f"    学习因子: {stats['learning_factor']:.3f}")
    
    # 测试调制因子
    print("\n2. 调制因子测试:")
    chem.reset()
    chem.state.dopamine = 0.8
    chem.state.serotonin = 0.3
    chem.state.norepinephrine = 0.6
    
    print(f"  学习率因子: {chem.get_learning_rate_factor():.3f}")
    print(f"  注意力提升: {chem.get_attention_boost():.3f}")
    print(f"  稳定性因子: {chem.get_stability_factor():.3f}")
    print(f"  可塑性因子: {chem.get_plasticity_factor():.3f}")
    print(f"  固化阈值: {chem.get_consolidation_threshold():.3f}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
