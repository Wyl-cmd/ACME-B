"""
脉冲细胞 (Spiking Cell) 实现
ACME的基本计算单元
"""

import numpy as np
from typing import Dict, Set, Optional, List
from dataclasses import dataclass, field


@dataclass
class Cell:
    """
    三态脉冲细胞
    
    状态:
        -1: 抑制 (inhibited)
         0: 静息 (resting)
        +1: 兴奋 (excited)
    
    属性:
        cell_id: 唯一标识符
        state: 当前状态
        potential: 膜电位
        energy: 能量值 (生命值)
        age: 年龄
        usefulness: 有用性评分
        threshold: 激活阈值 (自适应)
    """
    
    cell_id: int
    state: int = 0  # -1, 0, +1
    potential: float = 0.0
    energy: float = 1.0
    age: int = 0
    usefulness: float = 0.0
    threshold: float = 0.5
    
    # 连接
    synapses_out: Dict[int, 'Synapse'] = field(default_factory=dict)  # 输出连接
    synapses_in: Dict[int, 'Synapse'] = field(default_factory=dict)   # 输入连接
    
    # 脉冲历史 (用于STDP)
    last_spike_time: Optional[int] = None
    activation_history: List[int] = field(default_factory=list)
    
    # 保护期 (新细胞不易死亡)
    protection_period: int = 100
    
    # 不应期
    refractory_count: int = 0
    refractory_period: int = 5
    
    def __post_init__(self):
        """初始化后的设置"""
        self.initial_threshold = self.threshold
        self.activation_count = 0
    
    def update(self, chemical_field: 'ChemicalField', time_step: int) -> int:
        """
        细胞状态更新
        
        Args:
            chemical_field: 化学场 (全局调节)
            time_step: 当前时间步
            
        Returns:
            0: 无脉冲
            1: 产生脉冲
        """
        # 不应期递减
        if self.refractory_count > 0:
            self.refractory_count -= 1
            self.state = 0
            return 0
        
        # 收集输入
        total_input = self._collect_inputs()
        
        # 化学场调制 (去甲肾上腺素影响兴奋性)
        attention_boost = chemical_field.get_attention_boost()
        modulated_input = total_input * (0.5 + 0.5 * attention_boost)
        
        # 更新膜电位
        self.potential += modulated_input
        
        # 衰减
        self.potential *= 0.9
        
        # 状态转换
        fired = False
        
        if self.potential > self.threshold and self.state == 0:
            # 兴奋
            self._fire(time_step)
            fired = True
        elif self.potential < -self.threshold:
            # 抑制
            self.state = -1
        else:
            # 静息
            self.state = 0
        
        # 代谢
        self._metabolize(chemical_field)
        
        # 年龄增长
        self.age += 1
        
        return 1 if fired else 0
    
    def _collect_inputs(self) -> float:
        """收集来自上游细胞的输入"""
        total_input = 0.0
        
        for source_id, synapse in self.synapses_in.items():
            # 只考虑强连接 (稀疏性)
            if abs(synapse.weight) > 0.01:
                # 获取源细胞状态
                source_cell = synapse.pre_cell
                if source_cell.state == 1:  # 源细胞兴奋
                    total_input += synapse.weight * source_cell.state
        
        return total_input
    
    def _fire(self, time_step: int):
        """发射脉冲"""
        self.state = 1
        self.last_spike_time = time_step
        self.activation_history.append(time_step)
        self.activation_count += 1
        
        # 脉冲消耗能量
        self.energy -= 0.05
        
        # 设置不应期
        self.refractory_count = self.refractory_period
        
        # 传播到下游细胞
        self._propagate_spike()
    
    def _propagate_spike(self):
        """将脉冲传播到下游细胞"""
        for target_id, synapse in self.synapses_out.items():
            # 只传播强连接
            if abs(synapse.weight) > 0.01:
                target_cell = synapse.post_cell
                target_cell.potential += synapse.weight
    
    def _metabolize(self, chemical_field: 'ChemicalField'):
        """能量代谢"""
        # 基础消耗 (与连接数成正比)
        base_cost = 0.01 * (len(self.synapses_out) + len(self.synapses_in))
        
        # 有用细胞获得能量补贴
        if self.usefulness > 0.5:
            dopamine = chemical_field.dopamine
            energy_subsidy = dopamine * 0.02
        else:
            energy_subsidy = 0.0
        
        # 更新能量
        self.energy += energy_subsidy - base_cost
        
        # 限制范围
        self.energy = max(0.0, min(2.0, self.energy))
    
    def compute_usefulness(self, current_time: int, window: int = 1000) -> float:
        """
        计算有用性
        
        基于:
        - 近期激活频率
        - 对网络输出的贡献
        """
        if self.age < 10:  # 保护期
            return 0.5
        
        # 近期激活次数
        recent_activations = sum(
            1 for t in self.activation_history
            if current_time - t < window
        )
        
        # 激活频率
        activation_freq = recent_activations / window
        
        # 连接强度 (社会支持)
        connection_strength = sum(
            abs(s.weight) for s in self.synapses_out.values()
        ) + sum(
            abs(s.weight) for s in self.synapses_in.values()
        )
        
        # 综合有用性
        usefulness = activation_freq * 10 + connection_strength * 0.1
        
        self.usefulness = min(1.0, usefulness)
        return self.usefulness
    
    def add_synapse_out(self, synapse: 'Synapse'):
        """添加输出连接"""
        self.synapses_out[synapse.post_cell.cell_id] = synapse
    
    def add_synapse_in(self, synapse: 'Synapse'):
        """添加输入连接"""
        self.synapses_in[synapse.pre_cell.cell_id] = synapse
    
    def remove_synapse_out(self, target_id: int):
        """移除输出连接"""
        if target_id in self.synapses_out:
            del self.synapses_out[target_id]
    
    def remove_synapse_in(self, source_id: int):
        """移除输入连接"""
        if source_id in self.synapses_in:
            del self.synapses_in[source_id]
    
    def is_protected(self) -> bool:
        """检查是否处于保护期"""
        return self.age < self.protection_period
    
    def get_state_dict(self) -> dict:
        """获取状态字典 (用于保存)"""
        return {
            'cell_id': self.cell_id,
            'state': self.state,
            'potential': self.potential,
            'energy': self.energy,
            'age': self.age,
            'usefulness': self.usefulness,
            'threshold': self.threshold,
            'last_spike_time': self.last_spike_time,
            'activation_count': self.activation_count,
        }
    
    def __repr__(self):
        return f"Cell(id={self.cell_id}, state={self.state}, energy={self.energy:.2f}, age={self.age})"
