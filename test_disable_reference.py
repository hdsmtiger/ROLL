#!/usr/bin/env python3

"""
测试脚本：验证disable_reference功能是否正常工作
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from roll.configs.base_config import PPOConfig
from roll.pipeline.dpo.dpo_config import DPOConfig

def test_ppo_config_disable_reference():
    """测试PPOConfig中的disable_reference选项"""
    print("测试 PPOConfig disable_reference 功能...")
    
    # 测试默认情况（启用reference）
    config1 = PPOConfig(pretrain="test_model")
    assert config1.disable_reference is False, "默认情况下disable_reference应该为False"
    assert config1.reference.model_args.model_name_or_path == "test_model", "默认情况下reference应该初始化"
    print("✓ 默认情况下reference正常启用")
    
    # 测试禁用reference
    config2 = PPOConfig(pretrain="test_model", disable_reference=True)
    assert config2.disable_reference is True, "disable_reference应该为True"
    print("✓ disable_reference选项正常设置")
    
    print("PPOConfig 测试通过！\n")

def test_dpo_config_disable_reference():
    """测试DPOConfig中的disable_reference选项"""
    print("测试 DPOConfig disable_reference 功能...")
    
    # 测试默认情况（启用reference）
    config1 = DPOConfig(pretrain="test_model")
    assert config1.disable_reference is False, "默认情况下disable_reference应该为False"
    assert config1.reference.model_args.model_name_or_path == "test_model", "默认情况下reference应该初始化"
    print("✓ 默认情况下reference正常启用")
    
    # 测试禁用reference
    config2 = DPOConfig(pretrain="test_model", disable_reference=True)
    assert config2.disable_reference is True, "disable_reference应该为True"
    print("✓ disable_reference选项正常设置")
    
    print("DPOConfig 测试通过！\n")

if __name__ == "__main__":
    print("开始测试 disable_reference 功能...\n")
    
    try:
        test_ppo_config_disable_reference()
        test_dpo_config_disable_reference()
        print("🎉 所有测试通过！disable_reference 功能已成功实现。")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        sys.exit(1)