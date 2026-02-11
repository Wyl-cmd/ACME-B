"""
ACME: Autonomous Chemical-Morphological Evolution
一种仿生自进化的AI架构

核心特点:
- 脉冲神经网络 (Spiking Neural Network)
- STDP局部学习规则
- 化学场全局调节
- 生存压力驱动
- 动态结构演化
- 自主调控机制
"""

__version__ = "0.1.0"
__author__ = "ACME Research Team"

from .cell import Cell
from .synapse import Synapse, STDPLearning
from .chemical_field import ChemicalField
from .survival_system import SurvivalSystem
from .network import ACMENetwork
from .regulators import (
    DeathPreventionRegulator,
    CancerPreventionRegulator,
    MemoryConsolidationRegulator,
    StabilityRegulator,
    MetaRegulator
)

__all__ = [
    'Cell',
    'Synapse',
    'STDPLearning',
    'ChemicalField',
    'SurvivalSystem',
    'ACMENetwork',
    'DeathPreventionRegulator',
    'CancerPreventionRegulator',
    'MemoryConsolidationRegulator',
    'StabilityRegulator',
    'MetaRegulator',
]
