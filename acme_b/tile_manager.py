"""
ACME-B 瓦片管理器 (Tile Manager)
管理瓦片化稀疏计算和动态拓扑

核心功能:
- 瓦片分割与合并
- 稀疏性监测
- 负载均衡
- 动态拓扑演化 (第四纪元)
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class TileConfig:
    """瓦片配置"""
    tile_size: int = 64
    min_tile_size: int = 32
    max_tile_size: int = 128
    sparsity_target: float = 0.5
    growth_threshold: float = 0.8
    prune_threshold: float = 0.1


class TileManager:
    """
    瓦片管理器
    
    管理权重矩阵的瓦片化表示，支持:
    - 瓦片级别的稀疏性控制
    - 瓦片激活统计
    - 动态瓦片分裂/合并 (第四纪元)
    """
    
    def __init__(
        self,
        height: int,
        width: int,
        config: TileConfig = None
    ):
        self.height = height
        self.width = width
        self.config = config or TileConfig()
        
        self.tile_size = self.config.tile_size
        
        # 计算瓦片网格
        self.n_tiles_h = (height + self.tile_size - 1) // self.tile_size
        self.n_tiles_w = (width + self.tile_size - 1) // self.tile_size
        
        # 瓦片掩码 (0/1)
        self.tile_mask = torch.ones(
            self.n_tiles_h, 
            self.n_tiles_w,
            dtype=torch.float32
        )
        
        # 瓦片统计信息
        self.tile_activation_count = torch.zeros(
            self.n_tiles_h,
            self.n_tiles_w,
            dtype=torch.float32
        )
        
        self.tile_importance = torch.ones(
            self.n_tiles_h,
            self.n_tiles_w,
            dtype=torch.float32
        )
        
        # 瓦片年龄 (用于生命周期管理)
        self.tile_age = torch.zeros(
            self.n_tiles_h,
            self.n_tiles_w,
            dtype=torch.int32
        )
        
    def get_tile_indices(self, tile_h: int, tile_w: int) -> Tuple[int, int, int, int]:
        """
        获取瓦片在原始矩阵中的索引范围
        
        Returns:
            (h_start, h_end, w_start, w_end)
        """
        h_start = tile_h * self.tile_size
        h_end = min((tile_h + 1) * self.tile_size, self.height)
        w_start = tile_w * self.tile_size
        w_end = min((tile_w + 1) * self.tile_size, self.width)
        
        return h_start, h_end, w_start, w_end
    
    def expand_mask(self, tile_mask: torch.Tensor = None) -> torch.Tensor:
        """
        将瓦片掩码扩展到完整矩阵尺寸
        
        Args:
            tile_mask: 瓦片掩码 [n_tiles_h, n_tiles_w]
        
        Returns:
            full_mask: 完整掩码 [height, width]
        """
        if tile_mask is None:
            tile_mask = self.tile_mask
        
        full_mask = torch.zeros(
            self.height, 
            self.width,
            device=tile_mask.device
        )
        
        for i in range(self.n_tiles_h):
            for j in range(self.n_tiles_w):
                h_start, h_end, w_start, w_end = self.get_tile_indices(i, j)
                full_mask[h_start:h_end, w_start:w_end] = tile_mask[i, j]
        
        return full_mask
    
    def compress_mask(self, full_mask: torch.Tensor) -> torch.Tensor:
        """
        将完整掩码压缩为瓦片掩码
        
        Args:
            full_mask: 完整掩码 [height, width]
        
        Returns:
            tile_mask: 瓦片掩码 [n_tiles_h, n_tiles_w]
        """
        tile_mask = torch.zeros(
            self.n_tiles_h,
            self.n_tiles_w,
            device=full_mask.device
        )
        
        for i in range(self.n_tiles_h):
            for j in range(self.n_tiles_w):
                h_start, h_end, w_start, w_end = self.get_tile_indices(i, j)
                tile = full_mask[h_start:h_end, w_start:w_end]
                # 如果瓦片中有任何活跃元素，则瓦片活跃
                tile_mask[i, j] = (tile.sum() > 0).float()
        
        return tile_mask
    
    def update_activation_stats(self, activation_map: torch.Tensor):
        """
        更新瓦片激活统计
        
        Args:
            activation_map: 激活图 [height, width]
        """
        for i in range(self.n_tiles_h):
            for j in range(self.n_tiles_w):
                h_start, h_end, w_start, w_end = self.get_tile_indices(i, j)
                tile_activation = activation_map[h_start:h_end, w_start:w_end]
                
                # 累积激活次数
                self.tile_activation_count[i, j] += tile_activation.sum().item()
        
        # 更新年龄
        self.tile_age += 1
    
    def compute_tile_importance(self) -> torch.Tensor:
        """
        计算瓦片重要性
        
        基于:
        - 激活频率
        - 年龄 (新瓦片有保护)
        - 当前掩码状态
        """
        # 激活频率归一化
        if self.tile_activation_count.sum() > 0:
            activation_freq = self.tile_activation_count / self.tile_activation_count.sum()
        else:
            activation_freq = torch.zeros_like(self.tile_activation_count)
        
        # 年龄因子 (新瓦片重要性提升)
        age_factor = torch.exp(-self.tile_age.float() / 100)
        
        # 当前掩码
        mask_factor = self.tile_mask
        
        # 综合重要性
        importance = activation_freq * 0.5 + age_factor * 0.3 + mask_factor * 0.2
        
        self.tile_importance = importance
        return importance
    
    def get_sparse_tiles(self, sparsity: float = None) -> torch.Tensor:
        """
        根据稀疏度目标获取稀疏化后的掩码
        
        Args:
            sparsity: 目标稀疏度 (0-1)，None则使用配置值
        
        Returns:
            sparse_mask: 稀疏化后的掩码
        """
        if sparsity is None:
            sparsity = self.config.sparsity_target
        
        # 计算重要性
        importance = self.compute_tile_importance()
        
        # 根据重要性排序
        n_tiles = self.n_tiles_h * self.n_tiles_w
        n_keep = int(n_tiles * (1 - sparsity))
        
        # 保留最重要的n_keep个瓦片
        flat_importance = importance.view(-1)
        threshold = torch.topk(flat_importance, n_keep).values[-1]
        
        sparse_mask = (importance >= threshold).float()
        
        return sparse_mask
    
    def prune_tiles(self, prune_ratio: float = 0.1):
        """
        剪枝低重要性瓦片
        
        Args:
            prune_ratio: 剪枝比例
        """
        importance = self.compute_tile_importance()
        
        # 找出最不重要的瓦片
        n_prune = int(self.n_tiles_h * self.n_tiles_w * prune_ratio)
        flat_importance = importance.view(-1)
        _, prune_indices = torch.topk(flat_importance, n_prune, largest=False)
        
        # 剪枝 (设置掩码为0)
        flat_mask = self.tile_mask.view(-1)
        flat_mask[prune_indices] = 0
        self.tile_mask = flat_mask.view(self.n_tiles_h, self.n_tiles_w)
        
        print(f"Pruned {n_prune} tiles, sparsity: {self.get_sparsity():.2%}")
    
    def grow_tiles(self, grow_ratio: float = 0.1):
        """
        增长新瓦片 (第四纪元)
        
        在重要瓦片附近分裂新瓦片
        """
        importance = self.compute_tile_importance()
        
        # 找出高重要性瓦片
        high_importance_mask = importance > self.config.growth_threshold
        
        # 在这些瓦片周围激活新的瓦片
        new_mask = self.tile_mask.clone()
        
        for i in range(self.n_tiles_h):
            for j in range(self.n_tiles_w):
                if high_importance_mask[i, j] and self.tile_mask[i, j] > 0:
                    # 在周围激活新瓦片
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < self.n_tiles_h and 0 <= nj < self.n_tiles_w:
                                if self.tile_mask[ni, nj] == 0:
                                    # 以一定概率激活
                                    if torch.rand(1).item() < grow_ratio:
                                        new_mask[ni, nj] = 1
                                        self.tile_age[ni, nj] = 0  # 重置年龄
        
        n_new = (new_mask.sum() - self.tile_mask.sum()).item()
        self.tile_mask = new_mask
        
        print(f"Grew {n_new} new tiles, sparsity: {self.get_sparsity():.2%}")
    
    def get_sparsity(self) -> float:
        """获取当前稀疏度"""
        active_tiles = self.tile_mask.sum().item()
        total_tiles = self.n_tiles_h * self.n_tiles_w
        return 1 - (active_tiles / total_tiles)
    
    def get_active_tile_count(self) -> int:
        """获取活跃瓦片数"""
        return int(self.tile_mask.sum().item())
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            'n_tiles_h': self.n_tiles_h,
            'n_tiles_w': self.n_tiles_w,
            'total_tiles': self.n_tiles_h * self.n_tiles_w,
            'active_tiles': self.get_active_tile_count(),
            'sparsity': self.get_sparsity(),
            'avg_activation': self.tile_activation_count.mean().item(),
            'max_activation': self.tile_activation_count.max().item(),
        }


# 测试代码
if __name__ == '__main__':
    print("=" * 60)
    print("ACME-B 瓦片管理器测试")
    print("=" * 60)
    
    # 创建管理器
    manager = TileManager(768, 3072, TileConfig(tile_size=64))
    
    print(f"\n瓦片网格: {manager.n_tiles_h} x {manager.n_tiles_w}")
    print(f"总瓦片数: {manager.n_tiles_h * manager.n_tiles_w}")
    
    # 初始统计
    stats = manager.get_stats()
    print(f"\n初始状态:")
    print(f"  活跃瓦片: {stats['active_tiles']}/{stats['total_tiles']}")
    print(f"  稀疏度: {stats['sparsity']:.2%}")
    
    # 模拟激活
    print(f"\n模拟激活统计...")
    for _ in range(100):
        activation = torch.randn(768, 3072)
        manager.update_activation_stats(activation)
    
    # 计算重要性
    importance = manager.compute_tile_importance()
    print(f"  平均重要性: {importance.mean():.4f}")
    print(f"  最大重要性: {importance.max():.4f}")
    
    # 剪枝测试
    print(f"\n剪枝测试:")
    manager.prune_tiles(prune_ratio=0.2)
    print(f"  剪枝后稀疏度: {manager.get_sparsity():.2%}")
    
    # 增长测试
    print(f"\n增长测试:")
    manager.grow_tiles(grow_ratio=0.5)
    print(f"  增长后稀疏度: {manager.get_sparsity():.2%}")
    
    # 掩码扩展测试
    print(f"\n掩码扩展测试:")
    tile_mask = manager.get_sparse_tiles(sparsity=0.5)
    full_mask = manager.expand_mask(tile_mask)
    print(f"  瓦片掩码: {tile_mask.shape}")
    print(f"  完整掩码: {full_mask.shape}")
    print(f"  掩码匹配: {(manager.compress_mask(full_mask) == tile_mask).all()}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
