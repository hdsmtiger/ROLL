#!/usr/bin/env python3

"""
简化的单元测试：专门测试disable_reference配置功能
避免复杂的依赖问题，专注于核心配置逻辑
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch, MagicMock

def mock_dependencies():
    """Mock所有外部依赖"""
    # Mock torch
    mock_torch = Mock()
    mock_torch.cuda.device_count = Mock(return_value=8)
    sys.modules['torch'] = mock_torch
    
    # Mock transformers
    mock_transformers = Mock()
    sys.modules['transformers'] = mock_transformers
    
    # Mock其他依赖
    sys.modules['ray'] = Mock()
    sys.modules['datasets'] = Mock()
    sys.modules['codetiming'] = Mock()
    sys.modules['tqdm'] = Mock()
    sys.modules['numpy'] = Mock()
    sys.modules['omegaconf'] = Mock()
    
    # Mock platform
    mock_platform = Mock()
    mock_platform.device_count = Mock(return_value=8)
    
    return mock_platform

class TestDisableReferenceBasic(unittest.TestCase):
    """基础测试：disable_reference配置的核心功能"""
    
    def setUp(self):
        """设置测试环境"""
        mock_platform = mock_dependencies()
        
        with patch('roll.configs.base_config.current_platform', mock_platform):
            from roll.configs.base_config import PPOConfig
            from roll.pipeline.dpo.dpo_config import DPOConfig
            
            self.PPOConfig = PPOConfig
            self.DPOConfig = DPOConfig
    
    def test_ppo_config_has_disable_reference_field(self):
        """测试PPOConfig是否有disable_reference字段"""
        config = self.PPOConfig()
        self.assertTrue(hasattr(config, 'disable_reference'))
        self.assertIsInstance(config.disable_reference, bool)
    
    def test_ppo_config_disable_reference_default_value(self):
        """测试PPOConfig的disable_reference默认值"""
        config = self.PPOConfig()
        self.assertFalse(config.disable_reference)
    
    def test_ppo_config_disable_reference_set_true(self):
        """测试PPOConfig设置disable_reference=True"""
        config = self.PPOConfig(disable_reference=True)
        self.assertTrue(config.disable_reference)
    
    def test_dpo_config_has_disable_reference_field(self):
        """测试DPOConfig是否有disable_reference字段"""
        config = self.DPOConfig()
        self.assertTrue(hasattr(config, 'disable_reference'))
        self.assertIsInstance(config.disable_reference, bool)
    
    def test_dpo_config_disable_reference_default_value(self):
        """测试DPOConfig的disable_reference默认值"""
        config = self.DPOConfig()
        self.assertFalse(config.disable_reference)
    
    def test_dpo_config_disable_reference_set_true(self):
        """测试DPOConfig设置disable_reference=True"""
        config = self.DPOConfig(disable_reference=True)
        self.assertTrue(config.disable_reference)
    
    def test_ppo_config_reference_initialization_when_enabled(self):
        """测试PPOConfig在启用reference时的初始化"""
        config = self.PPOConfig(pretrain="test_model", disable_reference=False)
        self.assertFalse(config.disable_reference)
        self.assertEqual(config.reference.model_args.model_name_or_path, "test_model")
    
    def test_dpo_config_reference_initialization_when_enabled(self):
        """测试DPOConfig在启用reference时的初始化"""
        config = self.DPOConfig(pretrain="test_model", disable_reference=False)
        self.assertFalse(config.disable_reference)
        self.assertEqual(config.reference.model_args.model_name_or_path, "test_model")
    
    def test_config_field_type(self):
        """测试配置字段类型"""
        ppo_config = self.PPOConfig()
        dpo_config = self.DPOConfig()
        
        # 检查字段类型
        from roll.configs.base_config import dataclasses
        ppo_fields = dataclasses.fields(ppo_config)
        dpo_fields = dataclasses.fields(dpo_config)
        
        # 查找disable_reference字段
        ppo_disable_ref_field = next((f for f in ppo_fields if f.name == 'disable_reference'), None)
        dpo_disable_ref_field = next((f for f in dpo_fields if f.name == 'disable_reference'), None)
        
        self.assertIsNotNone(ppo_disable_ref_field)
        self.assertIsNotNone(dpo_disable_ref_field)
        self.assertEqual(ppo_disable_ref_field.type, bool)
        self.assertEqual(dpo_disable_ref_field.type, bool)


class TestDisableReferenceLogic(unittest.TestCase):
    """逻辑测试：disable_reference的相关逻辑"""
    
    def setUp(self):
        """设置测试环境"""
        mock_platform = mock_dependencies()
        
        with patch('roll.configs.base_config.current_platform', mock_platform):
            from roll.configs.base_config import PPOConfig
            from roll.pipeline.dpo.dpo_config import DPOConfig
            
            self.PPOConfig = PPOConfig
            self.DPOConfig = DPOConfig
    
    def test_ppo_config_post_init_logic(self):
        """测试PPOConfig的__post_init__逻辑"""
        # 测试启用reference的情况
        config_enabled = self.PPOConfig(pretrain="test_model", disable_reference=False)
        self.assertEqual(config_enabled.reference.model_args.model_name_or_path, "test_model")
        
        # 测试禁用reference的情况
        config_disabled = self.PPOConfig(pretrain="test_model", disable_reference=True)
        self.assertTrue(config_disabled.disable_reference)
    
    def test_dpo_config_post_init_logic(self):
        """测试DPOConfig的__post_init__逻辑"""
        # 测试启用reference的情况
        config_enabled = self.DPOConfig(pretrain="test_model", disable_reference=False)
        self.assertEqual(config_enabled.reference.model_args.model_name_or_path, "test_model")
        self.assertIsNotNone(config_enabled.reference.worker_cls)
        
        # 测试禁用reference的情况
        config_disabled = self.DPOConfig(pretrain="test_model", disable_reference=True)
        self.assertTrue(config_disabled.disable_reference)
        # 当禁用reference时，worker_cls应该为None
        self.assertIsNone(config_disabled.reference.worker_cls)


class TestFileContent(unittest.TestCase):
    """文件内容测试：验证代码修改是否正确"""
    
    def test_base_config_contains_disable_reference(self):
        """测试base_config.py包含disable_reference相关代码"""
        base_config_path = "/home/admin/iflow-cli-dev-service/iflow-workspace/ROLL/roll/configs/base_config.py"
        with open(base_config_path, 'r') as f:
            content = f.read()
        
        self.assertIn("disable_reference: bool", content)
        self.assertIn("not self.disable_reference", content)
        self.assertIn("Whether to disable the reference model", content)
    
    def test_dpo_config_contains_disable_reference(self):
        """测试dpo_config.py包含disable_reference相关代码"""
        dpo_config_path = "/home/admin/iflow-cli-dev-service/iflow-workspace/ROLL/roll/pipeline/dpo/dpo_config.py"
        with open(dpo_config_path, 'r') as f:
            content = f.read()
        
        self.assertIn("disable_reference: bool", content)
        self.assertIn("not self.disable_reference", content)
    
    def test_dpo_pipeline_contains_disable_reference(self):
        """测试dpo_pipeline.py包含disable_reference相关代码"""
        dpo_pipeline_path = "/home/admin/iflow-cli-dev-service/iflow-workspace/ROLL/roll/pipeline/dpo/dpo_pipeline.py"
        with open(dpo_pipeline_path, 'r') as f:
            content = f.read()
        
        self.assertIn("not self.pipeline_config.disable_reference", content)
        self.assertIn("When reference is disabled", content)
    
    def test_agentic_pipeline_contains_disable_reference(self):
        """测试agentic_pipeline.py包含disable_reference相关代码"""
        agentic_pipeline_path = "/home/admin/iflow-cli-dev-service/iflow-workspace/ROLL/roll/pipeline/agentic/agentic_pipeline.py"
        with open(agentic_pipeline_path, 'r') as f:
            content = f.read()
        
        self.assertIn("not self.pipeline_config.disable_reference", content)
        self.assertIn("When reference is disabled", content)


if __name__ == '__main__':
    print("开始运行disable_reference功能的单元测试...")
    print("=" * 60)
    
    # 运行测试
    unittest.main(verbosity=2)