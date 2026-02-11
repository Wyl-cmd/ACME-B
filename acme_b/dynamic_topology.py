"""
ACME-B 第四纪元: 动态拓扑演化

实现瓦片的完整生命周期管理:
- 瓦片分裂 (Tile Split)
- 瓦片凋亡 (Tile Apoptosis)
- 动态网络结构调整
- 资源自适应分配
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TileState:
    """瓦片状态"""
    energy: float = 1.0           # 能量值 [0, 2]
    usefulness: float = 0.5       # 有用性 [0, 1]
    age: int = 0                  # 年龄
    activation_count: int = 0     # 激活次数
    last_activation_time: int = 0 # 上次激活时间
    load_history: List[float] = None  # 负载历史
    
    def __post_init__(self):
        if self.load_history is None:
            self.load_history = []


class TileLifecycle:
    """
    瓦片生命周期管理器
    
    管理单个瓦片的生长、维持和死亡
    """
    
    def __init__(
        self,
        tile_id: Tuple[int, int],
        initial_energy: float = 1.0,
        protection_period: int = 100
    ):
        self.tile_id = tile_id
        self.state = TileState(energy=initial_energy)
        self.protection_period = protection_period
        
        # 分裂和凋亡阈值
        self.split_threshold = {
            'energy': 1.5,
            'usefulness': 0.7,
            'age_min': 50,
            'load': 0.8
        }
        
        self.apoptosis_threshold = {
            'energy': 0.1,
            'usefulness': 0.1,
            'age_max': 1000,
            'inactive_steps': 500
        }
    
    def update(
        self,
        current_load: float,
        global_performance: float,
        chemical_field,
        current_time: int
    ):
        """
        更新瓦片状态
        
        Args:
            current_load: 当前负载 [0, 1]
            global_performance: 全局性能 [0, 1]
            chemical_field: 化学场
            current_time: 当前时间步
        """
        # 记录负载
        self.state.load_history.append(current_load)
        if len(self.state.load_history) > 100:
            self.state.load_history.pop(0)
        
        # 更新有用性
        avg_load = np.mean(self.state.load_history) if self.state.load_history else 0
        self.state.usefulness = 0.7 * avg_load + 0.3 * global_performance
        
        # 能量代谢
        base_cost = 0.01 * (1 + current_load)
        
        # 有用性高的瓦片获得能量补贴
        if self.state.usefulness > 0.6:
            subsidy = chemical_field.state.dopamine * 0.02
        else:
            subsidy = 0
        
        self.state.energy += subsidy - base_cost
        self.state.energy = max(0.0, min(2.0, self.state.energy))
        
        # 更新激活统计
        if current_load > 0.1:
            self.state.activation_count += 1
            self.state.last_activation_time = current_time
        
        # 年龄增长
        self.state.age += 1
    
    def should_split(self, network_stats: Dict) -> bool:
        """
        判断是否应该分裂
        
        条件:
        1. 能量充足
        2. 有用性高
        3. 年龄足够
        4. 负载高
        5. 系统表现好
        6. 未达到容量上限
        """
        if self.state.age < self.protection_period:
            return False
        
        conditions = [
            self.state.energy >= self.split_threshold['energy'],
            self.state.usefulness >= self.split_threshold['usefulness'],
            self.state.age >= self.split_threshold['age_min'],
            np.mean(self.state.load_history[-10:]) >= self.split_threshold['load'] if len(self.state.load_history) >= 10 else False,
            network_stats.get('performance', 0) > 0.6,
            network_stats.get('tile_count', 0) < network_stats.get('capacity', 1000)
        ]
        
        return sum(conditions) >= 5  # 满足5/6条件
    
    def should_apoptosis(self, current_time: int) -> bool:
        """
        判断是否应该凋亡
        
        条件 (满足任一):
        1. 能量耗尽 (饿死)
        2. 长期无用且年老
        3. 长期不活跃
        4. 系统威胁下脆弱
        """
        if self.state.age < self.protection_period:
            return False
        
        # 饿死
        if self.state.energy <= self.apoptosis_threshold['energy']:
            return True
        
        # 无用且年老
        if (self.state.usefulness < self.apoptosis_threshold['usefulness'] and 
            self.state.age > self.apoptosis_threshold['age_max']):
            return True
        
        # 长期不活跃
        inactive_steps = current_time - self.state.last_activation_time
        if inactive_steps > self.apoptosis_threshold['inactive_steps']:
            return True
        
        return False
    
    def split(self) -> Tuple['TileLifecycle', 'TileLifecycle']:
        """
        瓦片分裂
        
        返回两个子瓦片
        """
        # 创建子瓦片
        child1 = TileLifecycle(
            tile_id=(self.tile_id[0], f"{self.tile_id[1]}_a"),
            initial_energy=self.state.energy * 0.4
        )
        
        child2 = TileLifecycle(
            tile_id=(self.tile_id[0], f"{self.tile_id[1]}_b"),
            initial_energy=self.state.energy * 0.4
        )
        
        # 父瓦片保留少量能量
        self.state.energy *= 0.2
        
        return child1, child2
    
    def get_stats(self) -> Dict:
        """获取瓦片统计"""
        return {
            'tile_id': self.tile_id,
            'energy': self.state.energy,
            'usefulness': self.state.usefulness,
            'age': self.state.age,
            'activation_count': self.state.activation_count,
            'avg_load': np.mean(self.state.load_history) if self.state.load_history else 0,
            'is_protected': self.state.age < self.protection_period
        }


class DynamicTopologyManager:
    """
    动态拓扑管理器
    
    管理整个网络的拓扑演化
    """
    
    def __init__(
        self,
        model: nn.Module,
        initial_capacity: int = 1000,
        growth_rate: float = 0.1,
        prune_rate: float = 0.05
    ):
        self.model = model
        self.capacity = initial_capacity
        self.growth_rate = growth_rate
        self.prune_rate = prune_rate
        
        # 瓦片生命周期管理
        self.tile_lifecycles: Dict[Tuple[int, int], TileLifecycle] = {}
        
        # 统计
        self.total_splits = 0
        self.total_apoptosis = 0
        self.evolution_history = []
    
    def register_tile(self, tile_id: Tuple[int, int]):
        """注册新瓦片"""
        if tile_id not in self.tile_lifecycles:
            self.tile_lifecycles[tile_id] = TileLifecycle(tile_id)
    
    def update_all_tiles(
        self,
        tile_loads: Dict[Tuple[int, int], float],
        global_performance: float,
        chemical_field,
        current_time: int
    ):
        """
        更新所有瓦片状态
        
        Args:
            tile_loads: 每个瓦片的负载 {tile_id: load}
            global_performance: 全局性能
            chemical_field: 化学场
            current_time: 当前时间
        """
        network_stats = {
            'performance': global_performance,
            'tile_count': len(self.tile_lifecycles),
            'capacity': self.capacity
        }
        
        for tile_id, lifecycle in self.tile_lifecycles.items():
            load = tile_loads.get(tile_id, 0.0)
            lifecycle.update(
                current_load=load,
                global_performance=global_performance,
                chemical_field=chemical_field,
                current_time=current_time
            )
    
    def evolve_topology(self, current_time: int) -> Dict:
        """
        执行拓扑演化
        
        返回演化统计
        """
        network_stats = {
            'performance': 0.7,  # 示例值
            'tile_count': len(self.tile_lifecycles),
            'capacity': self.capacity
        }
        
        splits = []
        apoptosis = []
        
        # 检查每个瓦片
        for tile_id, lifecycle in list(self.tile_lifecycles.items()):
            # 检查分裂
            if lifecycle.should_split(network_stats):
                child1, child2 = lifecycle.split()
                splits.append((tile_id, child1, child2))
                self.total_splits += 1
            
            # 检查凋亡
            elif lifecycle.should_apoptosis(current_time):
                apoptosis.append(tile_id)
                self.total_apoptosis += 1
        
        # 执行分裂
        for parent_id, child1, child2 in splits:
            self.tile_lifecycles[child1.tile_id] = child1
            self.tile_lifecycles[child2.tile_id] = child2
        
        # 执行凋亡
        for tile_id in apoptosis:
            if tile_id in self.tile_lifecycles:
                del self.tile_lifecycles[tile_id]
        
        # 记录历史
        self.evolution_history.append({
            'time': current_time,
            'splits': len(splits),
            'apoptosis': len(apoptosis),
            'total_tiles': len(self.tile_lifecycles)
        })
        
        return {
            'splits': len(splits),
            'apoptosis': len(apoptosis),
            'total_tiles': len(self.tile_lifecycles),
            'total_splits': self.total_splits,
            'total_apoptosis': self.total_apoptosis
        }
    
    def adjust_capacity(self, performance_trend: List[float]):
        """
        动态调整网络容量
        
        根据性能趋势调整承载力
        """
        if len(performance_trend) < 10:
            return
        
        recent_perf = np.mean(performance_trend[-10:])
        older_perf = np.mean(performance_trend[-20:-10]) if len(performance_trend) >= 20 else recent_perf
        
        trend = recent_perf - older_perf
        
        if trend > 0.05:  # 性能提升
            # 可以增加容量
            self.capacity = min(int(self.capacity * 1.1), 5000)
        elif trend < -0.05:  # 性能下降
            # 减少容量，集中资源
            self.capacity = max(int(self.capacity * 0.9), 100)
    
    def get_network_stats(self) -> Dict:
        """获取网络统计"""
        if not self.tile_lifecycles:
            return {'tile_count': 0}
        
        energies = [t.state.energy for t in self.tile_lifecycles.values()]
        usefulness = [t.state.usefulness for t in self.tile_lifecycles.values()]
        ages = [t.state.age for t in self.tile_lifecycles.values()]
        
        return {
            'tile_count': len(self.tile_lifecycles),
            'capacity': self.capacity,
            'avg_energy': np.mean(energies),
            'avg_usefulness': np.mean(usefulness),
            'avg_age': np.mean(ages),
            'total_splits': self.total_splits,
            'total_apoptosis': self.total_apoptosis,
            'utilization': len(self.tile_lifecycles) / self.capacity
        }


# 测试代码
if __name__ == '__main__':
    print("=" * 60)
    print("ACME-B 第四纪元: 动态拓扑演化测试")
    print("=" * 60)
    
    from .chemical_field import ChemicalField
    
    # 创建化学场
    chem = ChemicalField()
    
    # 创建拓扑管理器
    class DummyModel:
        pass
    
    manager = DynamicTopologyManager(DummyModel(), initial_capacity=100)
    
    # 注册一些瓦片
    print("\n1. 注册初始瓦片:")
    for i in range(5):
        for j in range(5):
            manager.register_tile((i, j))
    
    print(f"  初始瓦片数: {len(manager.tile_lifecycles)}")
    
    # 模拟演化
    print("\n2. 模拟拓扑演化:")
    
    for step in range(100):
        # 模拟负载
        tile_loads = {
            tile_id: np.random.uniform(0, 1) if np.random.rand() > 0.3 else 0
            for tile_id in manager.tile_lifecycles.keys()
        }
        
        # 更新所有瓦片
        manager.update_all_tiles(
            tile_loads=tile_loads,
            global_performance=0.7,
            chemical_field=chem,
            current_time=step
        )
        
        # 执行演化
        if step % 10 == 0:
            stats = manager.evolve_topology(step)
            if stats['splits'] > 0 or stats['apoptosis'] > 0:
                print(f"  Step {step}: splits={stats['splits']}, "
                      f"apoptosis={stats['apoptosis']}, "
                      f"total={stats['total_tiles']}")
    
    # 最终统计
    print("\n3. 最终统计:")
    final_stats = manager.get_network_stats()
    for key, value in final_stats.items():
        print(f"  {key}: {value}")
    
    # 测试单个瓦片生命周期
    print("\n4. 瓦片生命周期测试:")
    tile = TileLifecycle((0, 0))
    
    # 模拟高负载高表现
    for step in range(200):
        tile.update(
            current_load=0.9,
            global_performance=0.8,
            chemical_field=chem,
            current_time=step
        )
    
    print(f"  200步后能量: {tile.state.energy:.3f}")
    print(f"  有用性: {tile.state.usefulness:.3f}")
    print(f"  应该分裂? {tile.should_split({'performance': 0.8, 'tile_count': 50, 'capacity': 100})}")
    
    # 模拟低负载
    tile_low = TileLifecycle((1, 1))
    for step in range(200):
        tile_low.update(
            current_load=0.01,
            global_performance=0.5,
            chemical_field=chem,
            current_time=step
        )
    
    print(f"\n  低负载瓦片:")
    print(f"  200步后能量: {tile_low.state.energy:.3f}")
    print(f"  有用性: {tile_low.state.usefulness:.3f}")
    print(f"  应该凋亡? {tile_low.should_apoptosis(200)}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
