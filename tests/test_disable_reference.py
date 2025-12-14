#!/usr/bin/env python3

"""
disable_reference功能的单元测试
测试所有相关配置类和pipeline的disable_reference功能
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class TestDisableReferenceConfig(unittest.TestCase):
    """测试disable_reference配置功能"""
    
    def setUp(self):
        """测试前的准备工作"""
        self.test_pretrain = "test_model_path"
        
    def test_ppo_config_disable_reference_default(self):
        """测试PPOConfig默认情况下disable_reference为False"""
        # 模拟必要的依赖
        with patch('roll.configs.base_config.WorkerConfig'):
            with patch('roll.configs.base_config.logger'):
                from roll.configs.base_config import PPOConfig
                
                config = PPOConfig(pretrain=self.test_pretrain)
                self.assertFalse(config.disable_reference, "默认情况下disable_reference应该为False")
                
    def test_ppo_config_disable_reference_enabled(self):
        """测试PPOConfig启用disable_reference"""
        with patch('roll.configs.base_config.WorkerConfig'):
            with patch('roll.configs.base_config.logger'):
                from roll.configs.base_config import PPOConfig
                
                config = PPOConfig(pretrain=self.test_pretrain, disable_reference=True)
                self.assertTrue(config.disable_reference, "disable_reference应该为True")
                
    def test_ppo_config_post_init_with_disable_reference(self):
        """测试PPOConfig在disable_reference=True时的post_init行为"""
        with patch('roll.configs.base_config.WorkerConfig') as mock_worker_config:
            with patch('roll.configs.base_config.logger'):
                from roll.configs.base_config import PPOConfig
                
                # 创建mock worker config
                mock_ref_worker = Mock()
                mock_ref_worker.model_args.model_name_or_path = None
                mock_worker_config.return_value = mock_ref_worker
                
                config = PPOConfig(pretrain=self.test_pretrain, disable_reference=True)
                config.__post_init__()
                
                # 当disable_reference=True时，reference model path应该不被设置
                self.assertIsNone(mock_ref_worker.model_args.model_name_or_path, 
                                "disable_reference=True时不应该设置reference model path")
                
    def test_dpo_config_disable_reference_default(self):
        """测试DPOConfig默认情况下disable_reference为False"""
        with patch('roll.pipeline.dpo.dpo_config.WorkerConfig'):
            with patch('roll.pipeline.dpo.dpo_config.BaseConfig.__post_init__'):
                from roll.pipeline.dpo.dpo_config import DPOConfig
                
                config = DPOConfig(pretrain=self.test_pretrain)
                self.assertFalse(config.disable_reference, "默认情况下disable_reference应该为False")
                
    def test_dpo_config_disable_reference_enabled(self):
        """测试DPOConfig启用disable_reference"""
        with patch('roll.pipeline.dpo.dpo_config.WorkerConfig'):
            with patch('roll.pipeline.dpo.dpo_config.BaseConfig.__post_init__'):
                from roll.pipeline.dpo.dpo_config import DPOConfig
                
                config = DPOConfig(pretrain=self.test_pretrain, disable_reference=True)
                self.assertTrue(config.disable_reference, "disable_reference应该为True")
                
    def test_dpo_config_post_init_with_disable_reference(self):
        """测试DPOConfig在disable_reference=True时的post_init行为"""
        with patch('roll.pipeline.dpo.dpo_config.WorkerConfig') as mock_worker_config:
            with patch('roll.pipeline.dpo.dpo_config.BaseConfig.__post_init__'):
                from roll.pipeline.dpo.dpo_config import DPOConfig
                
                # 创建mock worker config
                mock_ref_worker = Mock()
                mock_ref_worker.model_args.model_name_or_path = None
                mock_ref_worker.worker_cls = None
                mock_worker_config.return_value = mock_ref_worker
                
                config = DPOConfig(pretrain=self.test_pretrain, disable_reference=True)
                config.__post_init__()
                
                # 当disable_reference=True时，reference model path应该不被设置
                self.assertIsNone(mock_ref_worker.model_args.model_name_or_path, 
                                "disable_reference=True时不应该设置reference model path")
                # worker_cls也不应该被设置
                self.assertIsNone(mock_ref_worker.worker_cls,
                                "disable_reference=True时不应该设置reference worker_cls")


class TestDisableReferencePipeline(unittest.TestCase):
    """测试disable_reference在pipeline中的功能"""
    
    def setUp(self):
        """测试前的准备工作"""
        self.test_config = Mock()
        self.test_config.disable_reference = False
        self.test_config.reference.name = "reference"
        self.test_config.reference.worker_cls = "test_worker_cls"
        
    @patch('roll.pipeline.dpo.dpo_pipeline.Cluster')
    @patch('roll.pipeline.dpo.dpo_pipeline.DataCollatorWithPaddingForDPO')
    @patch('roll.pipeline.dpo.dpo_pipeline.DataLoader')
    @patch('roll.pipeline.dpo.dpo_pipeline.datasets')
    @patch('roll.pipeline.dpo.dpo_pipeline.get_chat_template')
    @patch('roll.pipeline.dpo.dpo_pipeline.default_tokenizer_provider')
    def test_dpo_pipeline_reference_initialization(self, mock_tokenizer, mock_chat_template, 
                                                   mock_datasets, mock_dataloader, 
                                                   mock_collator, mock_cluster):
        """测试DPOPipeline中reference的初始化逻辑"""
        # 模拟所有依赖
        mock_tokenizer.return_value = Mock()
        mock_chat_template.return_value = Mock()
        mock_datasets.load_dataset.return_value = {"train": Mock()}
        mock_dataloader.return_value = Mock()
        mock_collator.return_value = Mock()
        
        with patch('roll.pipeline.dpo.dpo_pipeline.BasePipeline.__init__'):
            from roll.pipeline.dpo.dpo_pipeline import DPOPipeline
            
            # 测试disable_reference=False时的行为
            self.test_config.disable_reference = False
            pipeline = DPOPipeline(self.test_config)
            
            # 验证Cluster被调用
            self.assertTrue(mock_cluster.called, "disable_reference=False时应该创建reference cluster")
            
            # 重置mock
            mock_cluster.reset_mock()
            
            # 测试disable_reference=True时的行为
            self.test_config.disable_reference = True
            pipeline = DPOPipeline(self.test_config)
            
            # 验证reference为None
            self.assertIsNone(pipeline.reference, "disable_reference=True时reference应该为None")
            
    def test_agentic_config_disable_reference_logic(self):
        """测试AgenticConfig中disable_reference的逻辑"""
        with patch('roll.pipeline.agentic.agentic_config.PPOConfig.__post_init__'):
            with patch('roll.pipeline.agentic.agentic_config.EnvManagerConfig'):
                with patch('roll.pipeline.agentic.agentic_config.logger'):
                    from roll.pipeline.agentic.agentic_config import AgenticConfig
                    
                    # 创建mock reference worker
                    mock_ref_worker = Mock()
                    mock_ref_worker.worker_cls = None
                    
                    config = AgenticConfig(pretrain=self.test_pretrain, disable_reference=True)
                    config.reference = mock_ref_worker
                    config.__post_init__()
                    
                    # 当disable_reference=True时，reference worker_cls应该保持None
                    self.assertIsNone(mock_ref_worker.worker_cls,
                                    "disable_reference=True时不应该设置reference worker_cls")
                    
    def test_rlvr_config_disable_reference_logic(self):
        """测试RLVRConfig中disable_reference的逻辑"""
        with patch('roll.pipeline.rlvr.rlvr_config.PPOConfig.__post_init__'):
            with patch('roll.pipeline.rlvr.rlvr_config.logger'):
                from roll.pipeline.rlvr.rlvr_config import RLVRConfig
                
                # 创建mock reference worker
                mock_ref_worker = Mock()
                mock_ref_worker.worker_cls = None
                
                config = RLVRConfig(pretrain=self.test_pretrain, disable_reference=True)
                config.reference = mock_ref_worker
                config.__post_init__()
                
                # 当disable_reference=True时，reference worker_cls应该保持None
                self.assertIsNone(mock_ref_worker.worker_cls,
                                    "disable_reference=True时不应该设置reference worker_cls")


class TestDisableReferenceIntegration(unittest.TestCase):
    """测试disable_reference的集成功能"""
    
    def test_config_serialization(self):
        """测试配置的序列化功能"""
        with patch('roll.configs.base_config.WorkerConfig'):
            with patch('roll.configs.base_config.logger'):
                from roll.configs.base_config import PPOConfig
                
                config = PPOConfig(pretrain="test_model", disable_reference=True)
                config_dict = asdict(config)
                
                self.assertIn('disable_reference', config_dict, "配置字典应该包含disable_reference字段")
                self.assertTrue(config_dict['disable_reference'], "disable_reference应该为True")
                
    def test_config_inheritance(self):
        """测试配置继承关系"""
        with patch('roll.pipeline.dpo.dpo_config.WorkerConfig'):
            with patch('roll.pipeline.dpo.dpo_config.BaseConfig.__post_init__'):
                from roll.pipeline.dpo.dpo_config import DPOConfig
                from roll.configs.base_config import BaseConfig
                
                config = DPOConfig(pretrain="test_model", disable_reference=True)
                
                # 验证继承关系
                self.assertIsInstance(config, BaseConfig, "DPOConfig应该继承自BaseConfig")
                self.assertTrue(hasattr(config, 'disable_reference'), "DPOConfig应该有disable_reference属性")


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    loader = unittest.TestLoader()
    test_suite.addTest(loader.loadTestsFromTestCase(TestDisableReferenceConfig))
    test_suite.addTest(loader.loadTestsFromTestCase(TestDisableReferencePipeline))
    test_suite.addTest(loader.loadTestsFromTestCase(TestDisableReferenceIntegration))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 返回测试结果
    return result.wasSuccessful()


if __name__ == "__main__":
    print("开始运行disable_reference功能的单元测试...")
    print("=" * 60)
    
    success = run_tests()
    
    print("=" * 60)
    if success:
        print("🎉 所有单元测试通过！")
        sys.exit(0)
    else:
        print("❌ 部分单元测试失败！")
        sys.exit(1)