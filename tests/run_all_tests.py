"""
ACME-B 测试运行器
运行所有单元测试
"""

import unittest
import sys
import os

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入测试模块
from test_ternary_linear import TestTernaryLinear, TestACMEIntegration
from test_replay_buffer import TestReplayBuffer, TestFisherLock, TestIntegration


def run_all_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestTernaryLinear))
    suite.addTests(loader.loadTestsFromTestCase(TestACMEIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestReplayBuffer))
    suite.addTests(loader.loadTestsFromTestCase(TestFisherLock))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 返回结果
    return result.wasSuccessful()


if __name__ == '__main__':
    print("="*70)
    print("ACME-B 单元测试套件")
    print("="*70)
    print()
    
    success = run_all_tests()
    
    print()
    print("="*70)
    if success:
        print("所有测试通过！✓")
    else:
        print("部分测试失败！✗")
    print("="*70)
    
    sys.exit(0 if success else 1)
