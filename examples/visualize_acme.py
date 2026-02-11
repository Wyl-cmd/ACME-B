"""
ACME-B 训练可视化工具
实时监控训练过程、化学场状态、权重分布等
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
from collections import deque
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acme_b import ACMELinear, ChemicalField


class ACMEVisualizer:
    """ACME-B训练可视化器"""
    
    def __init__(self, model, update_interval=10):
        """
        Args:
            model: ACME-B模型
            update_interval: 更新间隔（批次）
        """
        self.model = model
        self.update_interval = update_interval
        self.batch_count = 0
        
        # 历史数据
        self.history = {
            'loss': deque(maxlen=1000),
            'accuracy': deque(maxlen=1000),
            'learning_rate': deque(maxlen=1000),
            'chemical_dopamine': deque(maxlen=500),
            'chemical_serotonin': deque(maxlen=500),
            'chemical_norepinephrine': deque(maxlen=500),
            'weight_sparsity': deque(maxlen=500),
            'tag_buffer_usage': deque(maxlen=500),
        }
        
        # 创建图形
        self.fig = None
        self.axes = None
        self._setup_plots()
    
    def _setup_plots(self):
        """设置绘图布局"""
        self.fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 3, figure=self.fig, hspace=0.3, wspace=0.3)
        
        # 1. 损失曲线
        self.ax_loss = self.fig.add_subplot(gs[0, :2])
        self.ax_loss.set_title('Training Loss', fontsize=12, fontweight='bold')
        self.ax_loss.set_xlabel('Batch')
        self.ax_loss.set_ylabel('Loss')
        self.ax_loss.grid(True, alpha=0.3)
        self.line_loss, = self.ax_loss.plot([], [], 'b-', linewidth=1, label='Loss')
        self.ax_loss.legend()
        
        # 2. 准确率曲线
        self.ax_acc = self.fig.add_subplot(gs[0, 2])
        self.ax_acc.set_title('Accuracy', fontsize=12, fontweight='bold')
        self.ax_acc.set_xlabel('Batch')
        self.ax_acc.set_ylabel('Accuracy')
        self.ax_acc.grid(True, alpha=0.3)
        self.line_acc, = self.ax_acc.plot([], [], 'g-', linewidth=1)
        
        # 3. 化学场状态
        self.ax_chemical = self.fig.add_subplot(gs[1, :2])
        self.ax_chemical.set_title('Chemical Field Modulation', fontsize=12, fontweight='bold')
        self.ax_chemical.set_xlabel('Batch')
        self.ax_chemical.set_ylabel('Level')
        self.ax_chemical.grid(True, alpha=0.3)
        self.line_dopamine, = self.ax_chemical.plot([], [], 'r-', linewidth=1.5, label='Dopamine (Reward)')
        self.line_serotonin, = self.ax_chemical.plot([], [], 'b-', linewidth=1.5, label='Serotonin (Stability)')
        self.line_norepinephrine, = self.ax_chemical.plot([], [], 'g-', linewidth=1.5, label='Norepinephrine (Focus)')
        self.ax_chemical.legend(loc='upper right', fontsize=8)
        self.ax_chemical.set_ylim([0, 2])
        
        # 4. 权重分布
        self.ax_weight_dist = self.fig.add_subplot(gs[1, 2])
        self.ax_weight_dist.set_title('Weight Distribution', fontsize=12, fontweight='bold')
        self.ax_weight_dist.set_xlabel('Weight Value')
        self.ax_weight_dist.set_ylabel('Count')
        
        # 5. 权重稀疏度
        self.ax_sparsity = self.fig.add_subplot(gs[2, 0])
        self.ax_sparsity.set_title('Weight Sparsity', fontsize=12, fontweight='bold')
        self.ax_sparsity.set_xlabel('Batch')
        self.ax_sparsity.set_ylabel('Sparsity (%)')
        self.ax_sparsity.grid(True, alpha=0.3)
        self.line_sparsity, = self.ax_sparsity.plot([], [], 'purple', linewidth=1.5)
        
        # 6. Tag Buffer使用情况
        self.ax_tag_usage = self.fig.add_subplot(gs[2, 1])
        self.ax_tag_usage.set_title('Tag Buffer Usage', fontsize=12, fontweight='bold')
        self.ax_tag_usage.set_xlabel('Batch')
        self.ax_tag_usage.set_ylabel('Usage (%)')
        self.ax_tag_usage.grid(True, alpha=0.3)
        self.line_tag_usage, = self.ax_tag_usage.plot([], [], 'orange', linewidth=1.5)
        
        # 7. 层激活热力图
        self.ax_activation = self.fig.add_subplot(gs[2, 2])
        self.ax_activation.set_title('Layer Activations', fontsize=12, fontweight='bold')
        self.ax_activation.axis('off')
        
        plt.tight_layout()
        plt.ion()  # 开启交互模式
    
    def update(self, loss=None, accuracy=None, learning_rate=None):
        """更新可视化"""
        self.batch_count += 1
        
        # 记录数据
        if loss is not None:
            self.history['loss'].append(loss)
        if accuracy is not None:
            self.history['accuracy'].append(accuracy)
        if learning_rate is not None:
            self.history['learning_rate'].append(learning_rate)
        
        # 获取化学场状态
        if hasattr(self.model, 'chemical_field') and self.model.chemical_field is not None:
            cf = self.model.chemical_field
            self.history['chemical_dopamine'].append(cf.dopamine_level)
            self.history['chemical_serotonin'].append(cf.serotonin_level)
            self.history['chemical_norepinephrine'].append(cf.norepinephrine_level)
        
        # 计算权重稀疏度
        sparsity = self._compute_weight_sparsity()
        self.history['weight_sparsity'].append(sparsity)
        
        # 计算Tag Buffer使用率
        tag_usage = self._compute_tag_buffer_usage()
        self.history['tag_buffer_usage'].append(tag_usage)
        
        # 定期更新图表
        if self.batch_count % self.update_interval == 0:
            self._draw()
    
    def _compute_weight_sparsity(self):
        """计算权重稀疏度（0值比例）"""
        total_params = 0
        zero_params = 0
        
        for module in self.model.modules():
            if isinstance(module, ACMELinear):
                if hasattr(module, 'weight_base'):
                    weights = module.weight_base.data
                    total_params += weights.numel()
                    zero_params += (weights == 0).sum().item()
        
        if total_params == 0:
            return 0
        return (zero_params / total_params) * 100
    
    def _compute_tag_buffer_usage(self):
        """计算Tag Buffer使用率"""
        total_capacity = 0
        used_capacity = 0
        
        for module in self.model.modules():
            if isinstance(module, ACMELinear):
                if hasattr(module, 'tag_buffer') and module.tag_buffer is not None:
                    total_capacity += module.tag_buffer.capacity
                    used_capacity += module.tag_buffer.size
        
        if total_capacity == 0:
            return 0
        return (used_capacity / total_capacity) * 100
    
    def _draw(self):
        """绘制图表"""
        # 1. 更新损失曲线
        if len(self.history['loss']) > 0:
            x = range(len(self.history['loss']))
            self.line_loss.set_data(x, self.history['loss'])
            self.ax_loss.relim()
            self.ax_loss.autoscale_view()
        
        # 2. 更新准确率曲线
        if len(self.history['accuracy']) > 0:
            x = range(len(self.history['accuracy']))
            self.line_acc.set_data(x, self.history['accuracy'])
            self.ax_acc.relim()
            self.ax_acc.autoscale_view()
        
        # 3. 更新化学场
        if len(self.history['chemical_dopamine']) > 0:
            x = range(len(self.history['chemical_dopamine']))
            self.line_dopamine.set_data(x, self.history['chemical_dopamine'])
            self.line_serotonin.set_data(x, self.history['chemical_serotonin'])
            self.line_norepinephrine.set_data(x, self.history['chemical_norepinephrine'])
            self.ax_chemical.relim()
            self.ax_chemical.autoscale_view()
        
        # 4. 更新权重分布直方图
        self.ax_weight_dist.clear()
        self.ax_weight_dist.set_title('Weight Distribution', fontsize=12, fontweight='bold')
        weights = self._get_all_weights()
        if len(weights) > 0:
            self.ax_weight_dist.hist(weights, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
            self.ax_weight_dist.set_xlabel('Weight Value')
            self.ax_weight_dist.set_ylabel('Count')
            self.ax_weight_dist.axvline(x=0, color='red', linestyle='--', linewidth=1, label='Zero')
        
        # 5. 更新稀疏度
        if len(self.history['weight_sparsity']) > 0:
            x = range(len(self.history['weight_sparsity']))
            self.line_sparsity.set_data(x, self.history['weight_sparsity'])
            self.ax_sparsity.relim()
            self.ax_sparsity.autoscale_view()
        
        # 6. 更新Tag Buffer使用率
        if len(self.history['tag_buffer_usage']) > 0:
            x = range(len(self.history['tag_buffer_usage']))
            self.line_tag_usage.set_data(x, self.history['tag_buffer_usage'])
            self.ax_tag_usage.relim()
            self.ax_tag_usage.autoscale_view()
        
        # 7. 更新层激活热力图
        self._draw_activation_heatmap()
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)
    
    def _get_all_weights(self):
        """获取所有权重值"""
        all_weights = []
        for module in self.model.modules():
            if isinstance(module, ACMELinear):
                if hasattr(module, 'weight_base'):
                    weights = module.weight_base.data.cpu().numpy().flatten()
                    all_weights.extend(weights)
        return np.array(all_weights)
    
    def _draw_activation_heatmap(self):
        """绘制层激活热力图"""
        self.ax_activation.clear()
        self.ax_activation.set_title('Layer Activations', fontsize=12, fontweight='bold')
        self.ax_activation.axis('off')
        
        # 这里可以添加实际的激活值可视化
        # 目前显示一个占位图
        placeholder = np.random.rand(10, 10)
        im = self.ax_activation.imshow(placeholder, cmap='hot', aspect='auto')
        self.ax_activation.set_xticks([])
        self.ax_activation.set_yticks([])
    
    def save(self, path):
        """保存当前图表"""
        self.fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"可视化图表已保存到: {path}")
    
    def close(self):
        """关闭可视化"""
        plt.ioff()
        plt.close(self.fig)


class TopologyVisualizer:
    """拓扑演化可视化器"""
    
    def __init__(self, topology_manager):
        """
        Args:
            topology_manager: DynamicTopologyManager实例
        """
        self.topology_manager = topology_manager
        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 10))
        self.fig.suptitle('ACME-B Topology Evolution', fontsize=14, fontweight='bold')
        
    def update(self):
        """更新拓扑可视化"""
        # 1. Tile数量变化
        ax = self.axes[0, 0]
        ax.clear()
        ax.set_title('Tile Count Over Time', fontsize=11, fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Number of Tiles')
        ax.grid(True, alpha=0.3)
        
        if len(self.topology_manager.history) > 0:
            steps = [h['step'] for h in self.topology_manager.history]
            counts = [h['num_tiles'] for h in self.topology_manager.history]
            ax.plot(steps, counts, 'b-', linewidth=2, marker='o', markersize=4)
        
        # 2. 平均能量
        ax = self.axes[0, 1]
        ax.clear()
        ax.set_title('Average Tile Energy', fontsize=11, fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Energy')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, color='r', linestyle='--', label='Equilibrium')
        
        if len(self.topology_manager.history) > 0:
            steps = [h['step'] for h in self.topology_manager.history]
            energies = [h['avg_energy'] for h in self.topology_manager.history]
            ax.plot(steps, energies, 'g-', linewidth=2, marker='s', markersize=4)
            ax.legend()
        
        # 3. 分裂/凋亡事件
        ax = self.axes[1, 0]
        ax.clear()
        ax.set_title('Lifecycle Events', fontsize=11, fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Cumulative Events')
        ax.grid(True, alpha=0.3)
        
        if len(self.topology_manager.history) > 0:
            steps = [h['step'] for h in self.topology_manager.history]
            splits = [h['num_splits'] for h in self.topology_manager.history]
            deaths = [h['num_deaths'] for h in self.topology_manager.history]
            ax.plot(steps, splits, 'g-', linewidth=2, marker='^', markersize=5, label='Splits')
            ax.plot(steps, deaths, 'r-', linewidth=2, marker='v', markersize=5, label='Deaths')
            ax.legend()
        
        # 4. Tile状态分布
        ax = self.axes[1, 1]
        ax.clear()
        ax.set_title('Current Tile States', fontsize=11, fontweight='bold')
        
        if len(self.topology_manager.tiles) > 0:
            energies = [t.state.energy for t in self.topology_manager.tiles.values()]
            usefulness = [t.state.usefulness for t in self.topology_manager.tiles.values()]
            ages = [t.state.age for t in self.topology_manager.tiles.values()]
            
            # 散点图：能量 vs 有用性
            scatter = ax.scatter(energies, usefulness, c=ages, cmap='viridis', 
                               s=100, alpha=0.6, edgecolors='black')
            ax.set_xlabel('Energy')
            ax.set_ylabel('Usefulness')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0.7, color='g', linestyle='--', alpha=0.5, label='Split threshold')
            ax.axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='Death threshold')
            ax.axvline(x=1.5, color='g', linestyle='--', alpha=0.5)
            ax.axvline(x=0.1, color='r', linestyle='--', alpha=0.5)
            plt.colorbar(scatter, ax=ax, label='Age')
        
        plt.tight_layout()
        plt.pause(0.01)
    
    def save(self, path):
        """保存图表"""
        self.fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"拓扑可视化已保存到: {path}")


def visualize_weight_matrix(model, layer_idx=0, save_path=None):
    """
    可视化权重矩阵
    
    Args:
        model: ACME-B模型
        layer_idx: 层索引
        save_path: 保存路径
    """
    # 找到指定的ACMELinear层
    acme_layers = [m for m in model.modules() if isinstance(m, ACMELinear)]
    if layer_idx >= len(acme_layers):
        print(f"Layer index {layer_idx} out of range")
        return
    
    layer = acme_layers[layer_idx]
    
    # 获取权重
    if hasattr(layer, 'weight_base'):
        weights = layer.weight_base.data.cpu().numpy()
    else:
        print("Layer has no weight_base")
        return
    
    # 创建可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Weight Matrix Visualization (Layer {layer_idx})', fontsize=14, fontweight='bold')
    
    # 1. 原始权重热力图
    ax = axes[0]
    im = ax.imshow(weights, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax.set_title('Weight Heatmap', fontsize=11, fontweight='bold')
    ax.set_xlabel('Input Dimension')
    ax.set_ylabel('Output Dimension')
    plt.colorbar(im, ax=ax)
    
    # 2. 三值化后的权重
    ax = axes[1]
    weights_ternary = np.sign(weights)  # 简化的三值化显示
    weights_ternary[weights_ternary == 0] = 0
    im = ax.imshow(weights_ternary, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax.set_title('Ternarized Weights', fontsize=11, fontweight='bold')
    ax.set_xlabel('Input Dimension')
    ax.set_ylabel('Output Dimension')
    plt.colorbar(im, ax=ax)
    
    # 3. 权重分布
    ax = axes[2]
    ax.hist(weights.flatten(), bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(x=-1, color='red', linestyle='--', linewidth=2, label='-1')
    ax.axvline(x=0, color='green', linestyle='--', linewidth=2, label='0')
    ax.axvline(x=1, color='red', linestyle='--', linewidth=2, label='+1')
    ax.set_title('Weight Distribution', fontsize=11, fontweight='bold')
    ax.set_xlabel('Weight Value')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"权重可视化已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_chemical_field_timeline(history, save_path=None):
    """
    可视化化学场时间线
    
    Args:
        history: 化学场历史数据列表
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Chemical Field Timeline', fontsize=14, fontweight='bold')
    
    steps = range(len(history))
    
    # 1. 多巴胺
    ax = axes[0, 0]
    dopamine = [h['dopamine'] for h in history]
    ax.plot(steps, dopamine, 'r-', linewidth=2)
    ax.fill_between(steps, dopamine, alpha=0.3, color='red')
    ax.set_title('Dopamine (Reward/Pleasure)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Level')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Baseline')
    ax.legend()
    
    # 2. 血清素
    ax = axes[0, 1]
    serotonin = [h['serotonin'] for h in history]
    ax.plot(steps, serotonin, 'b-', linewidth=2)
    ax.fill_between(steps, serotonin, alpha=0.3, color='blue')
    ax.set_title('Serotonin (Stability/Mood)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Level')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Baseline')
    ax.legend()
    
    # 3. 去甲肾上腺素
    ax = axes[1, 0]
    norepinephrine = [h['norepinephrine'] for h in history]
    ax.plot(steps, norepinephrine, 'g-', linewidth=2)
    ax.fill_between(steps, norepinephrine, alpha=0.3, color='green')
    ax.set_title('Norepinephrine (Focus/Arousal)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Level')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Baseline')
    ax.legend()
    
    # 4. 综合视图
    ax = axes[1, 1]
    ax.plot(steps, dopamine, 'r-', linewidth=2, label='Dopamine', alpha=0.8)
    ax.plot(steps, serotonin, 'b-', linewidth=2, label='Serotonin', alpha=0.8)
    ax.plot(steps, norepinephrine, 'g-', linewidth=2, label='Norepinephrine', alpha=0.8)
    ax.set_title('Combined View', fontsize=11, fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Level')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"化学场时间线已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_training_report(model, history, save_dir='./training_report'):
    """
    创建完整的训练报告
    
    Args:
        model: 训练后的模型
        history: 训练历史
        save_dir: 保存目录
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n生成训练报告到: {save_dir}")
    
    # 1. 权重可视化
    visualize_weight_matrix(model, layer_idx=0, 
                           save_path=os.path.join(save_dir, 'weights_layer0.png'))
    
    # 2. 化学场时间线
    if 'chemical' in history:
        visualize_chemical_field_timeline(history['chemical'], 
                                         save_path=os.path.join(save_dir, 'chemical_timeline.png'))
    
    # 3. 训练曲线
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Training Report', fontsize=14, fontweight='bold')
    
    # 损失曲线
    if 'loss' in history:
        ax = axes[0, 0]
        ax.plot(history['loss'], 'b-', linewidth=1.5)
        ax.set_title('Training Loss', fontsize=11, fontweight='bold')
        ax.set_xlabel('Batch')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
    
    # 准确率曲线
    if 'accuracy' in history:
        ax = axes[0, 1]
        ax.plot(history['accuracy'], 'g-', linewidth=1.5)
        ax.set_title('Training Accuracy', fontsize=11, fontweight='bold')
        ax.set_xlabel('Batch')
        ax.set_ylabel('Accuracy')
        ax.grid(True, alpha=0.3)
    
    # 学习率
    if 'learning_rate' in history:
        ax = axes[1, 0]
        ax.plot(history['learning_rate'], 'purple', linewidth=1.5)
        ax.set_title('Learning Rate', fontsize=11, fontweight='bold')
        ax.set_xlabel('Batch')
        ax.set_ylabel('LR')
        ax.grid(True, alpha=0.3)
    
    # 稀疏度
    if 'sparsity' in history:
        ax = axes[1, 1]
        ax.plot(history['sparsity'], 'orange', linewidth=1.5)
        ax.set_title('Weight Sparsity', fontsize=11, fontweight='bold')
        ax.set_xlabel('Batch')
        ax.set_ylabel('Sparsity (%)')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. 模型统计
    stats = {
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    
    if hasattr(model, 'get_memory_usage'):
        stats['memory_usage_mb'] = model.get_memory_usage() / 1024 / 1024
    
    # 保存统计信息
    with open(os.path.join(save_dir, 'stats.txt'), 'w') as f:
        f.write("ACME-B Training Report\n")
        f.write("=" * 50 + "\n\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    print(f"报告生成完成！")
    print(f"  - 权重可视化: {save_dir}/weights_layer0.png")
    print(f"  - 训练曲线: {save_dir}/training_curves.png")
    print(f"  - 统计信息: {save_dir}/stats.txt")


# 示例用法
if __name__ == "__main__":
    print("ACME-B 可视化工具")
    print("=" * 50)
    print("\n使用方法:")
    print("1. 在训练脚本中导入: from visualize_acme import ACMEVisualizer")
    print("2. 创建可视化器: visualizer = ACMEVisualizer(model)")
    print("3. 在每个batch后更新: visualizer.update(loss, accuracy)")
    print("4. 训练结束后保存: visualizer.save('training.png')")
    print("\n其他功能:")
    print("- visualize_weight_matrix: 可视化权重矩阵")
    print("- visualize_chemical_field_timeline: 可视化化学场时间线")
    print("- create_training_report: 创建完整训练报告")
