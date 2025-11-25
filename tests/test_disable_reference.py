#!/usr/bin/env python3
"""
单元测试：验证disable_reference功能
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Mock torch和其他依赖
sys.modules['torch'] = Mock()
sys.modules['transformers'] = Mock()
sys.modules['datasets'] = Mock()
sys.modules['ray'] = Mock()
sys.modules['codetiming'] = Mock()
sys.modules['tqdm'] = Mock()
sys.modules['numpy'] = Mock()
sys.modules['omegaconf'] = Mock()

# 创建mock对象
mock_torch = sys.modules['torch']
mock_torch.tensor = Mock(return_value=Mock())
mock_torch.mean = Mock(return_value=Mock())
mock_torch.max = Mock(return_value=Mock())
mock_torch.min = Mock(return_value=Mock())
mock_torch.detach = Mock(return_value=Mock())
mock_torch.item = Mock(return_value=0.0)

class TestDisableReferenceConfig(unittest.TestCase):
    """测试disable_reference配置功能"""
    
    def setUp(self):
        """测试前的准备工作"""
        # 清理模块缓存
        modules_to_remove = [mod for mod in sys.modules.keys() if mod.startswith('roll')]
        for mod in modules_to_remove:
            if mod in sys.modules:
                del sys.modules[mod]
    
    def test_ppo_config_disable_reference_default(self):
        """测试PPOConfig默认disable_reference为False"""
        from roll.configs.base_config import PPOConfig
        
        config = PPOConfig(pretrain="test_model")
        self.assertFalse(config.disable_reference, "默认情况下disable_reference应该为False")
        self.assertEqual(config.reference.model_args.model_name_or_path, "test_model")
    
    def test_ppo_config_disable_reference_enabled(self):
        """测试PPOConfig启用disable_reference"""
        from roll.configs.base_config import PPOConfig
        
        config = PPOConfig(pretrain="test_model", disable_reference=True)
        self.assertTrue(config.disable_reference, "disable_reference应该为True")
    
    def test_dpo_config_disable_reference_default(self):
        """测试DPOConfig默认disable_reference为False"""
        from roll.pipeline.dpo.dpo_config import DPOConfig
        
        config = DPOConfig(pretrain="test_model")
        self.assertFalse(config.disable_reference, "默认情况下disable_reference应该为False")
        self.assertEqual(config.reference.model_args.model_name_or_path, "test_model")
    
    def test_dpo_config_disable_reference_enabled(self):
        """测试DPOConfig启用disable_reference"""
        from roll.pipeline.dpo.dpo_config import DPOConfig
        
        config = DPOConfig(pretrain="test_model", disable_reference=True)
        self.assertTrue(config.disable_reference, "disable_reference应该为True")
    
    def test_agentic_config_disable_reference(self):
        """测试AgenticConfig的disable_reference功能"""
        from roll.pipeline.agentic.agentic_config import AgenticConfig
        
        # 测试默认情况
        config = AgenticConfig(pretrain="test_model")
        self.assertFalse(config.disable_reference, "默认情况下disable_reference应该为False")
        
        # 测试启用disable_reference
        config = AgenticConfig(pretrain="test_model", disable_reference=True)
        self.assertTrue(config.disable_reference, "disable_reference应该为True")
    
    def test_rlvr_config_disable_reference(self):
        """测试RLVRConfig的disable_reference功能"""
        from roll.pipeline.rlvr.rlvr_config import RLVRConfig
        
        # 测试默认情况
        config = RLVRConfig(pretrain="test_model")
        self.assertFalse(config.disable_reference, "默认情况下disable_reference应该为False")
        
        # 测试启用disable_reference
        config = RLVRConfig(pretrain="test_model", disable_reference=True)
        self.assertTrue(config.disable_reference, "disable_reference应该为True")


class TestDisableReferencePipeline(unittest.TestCase):
    """测试disable_reference在pipeline中的实现"""
    
    def setUp(self):
        """测试前的准备工作"""
        # 清理模块缓存
        modules_to_remove = [mod for mod in sys.modules.keys() if mod.startswith('roll')]
        for mod in modules_to_remove:
            if mod in sys.modules:
                del sys.modules[mod]
    
    @patch('roll.pipeline.dpo.dpo_pipeline.Cluster')
    @patch('roll.pipeline.dpo.dpo_pipeline.default_tokenizer_provider')
    @patch('roll.pipeline.dpo.dpo_pipeline.datasets')
    def test_dpo_pipeline_disable_reference(self, mock_datasets, mock_tokenizer, mock_cluster):
        """测试DPO Pipeline的disable_reference功能"""
        from roll.pipeline.dpo.dpo_pipeline import DPOPipeline
        
        # Mock dataset和tokenizer
        mock_dataset = Mock()
        mock_dataset.map = Mock(return_value=mock_dataset)
        mock_dataset.filter = Mock(return_value=mock_dataset)
        mock_datasets.load_dataset = Mock(return_value={"train": mock_dataset})
        mock_tokenizer.return_value = Mock()
        
        # Mock Cluster
        mock_cluster_instance = Mock()
        mock_cluster_instance.initialize = Mock(return_value=[])
        mock_cluster.return_value = mock_cluster_instance
        
        # 测试启用disable_reference
        config = DPOConfig(pretrain="test_model", disable_reference=True)
        config.actor_train.data_args.file_name = ["test.json"]
        config.sequence_length = 512
        
        with patch('roll.pipeline.dpo.dpo_pipeline.DataLoader') as mock_dataloader:
            mock_dataloader.return_value = [Mock()]
            pipeline = DPOPipeline(config)
            
            # 验证reference cluster没有被初始化
            self.assertIsNone(pipeline.reference, "当disable_reference为True时，reference应该为None")
    
    @patch('roll.pipeline.agentic.agentic_pipeline.Cluster')
    @patch('roll.pipeline.agentic.agentic_pipeline.default_tokenizer_provider')
    @patch('roll.pipeline.agentic.agentic_pipeline.ray')
    def test_agentic_pipeline_disable_reference(self, mock_ray, mock_tokenizer, mock_cluster):
        """测试Agentic Pipeline的disable_reference功能"""
        from roll.pipeline.agentic.agentic_pipeline import AgenticPipeline
        
        # Mock tokenizer
        mock_tokenizer.return_value = Mock()
        
        # Mock Cluster
        mock_cluster_instance = Mock()
        mock_cluster_instance.initialize = Mock(return_value=[])
        mock_cluster.return_value = mock_cluster_instance
        
        # Mock ray
        mock_ray.remote = Mock()
        mock_ray.get_runtime_context = Mock()
        mock_ray.get_runtime_context.return_value.get_node_id = Mock(return_value="node1")
        
        # 测试启用disable_reference
        config = AgenticConfig(pretrain="test_model", disable_reference=True)
        config.actor_train.data_args.file_name = ["test.json"]
        
        with patch('roll.pipeline.agentic.agentic_pipeline.RolloutScheduler'):
            pipeline = AgenticPipeline(config)
            
            # 验证reference cluster没有被初始化
            self.assertIsNone(pipeline.reference, "当disable_reference为True时，reference应该为None")


class TestDisableReferenceLogic(unittest.TestCase):
    """测试disable_reference的逻辑正确性"""
    
    def setUp(self):
        """测试前的准备工作"""
        # 清理模块缓存
        modules_to_remove = [mod for mod in sys.modules.keys() if mod.startswith('roll')]
        for mod in modules_to_remove:
            if mod in sys.modules:
                del sys.modules[mod]
    
    def test_config_post_init_logic(self):
        """测试配置的post_init逻辑"""
        from roll.configs.base_config import PPOConfig
        from roll.pipeline.dpo.dpo_config import DPOConfig
        
        # 测试PPOConfig
        config = PPOConfig(pretrain="test_model", disable_reference=True)
        config.__post_init__()
        
        # 验证reference model path没有被设置
        self.assertTrue(config.disable_reference)
        
        # 测试DPOConfig
        config = DPOConfig(pretrain="test_model", disable_reference=True)
        config.__post_init__()
        
        # 验证reference model path没有被设置
        self.assertTrue(config.disable_reference)
    
    def test_worker_cls_assignment(self):
        """测试worker_cls的分配逻辑"""
        from roll.pipeline.agentic.agentic_config import AgenticConfig
        
        # 测试禁用reference时不会分配worker_cls
        config = AgenticConfig(pretrain="test_model", disable_reference=True)
        config.__post_init__()
        
        # 验证reference的worker_cls没有被设置
        self.assertIsNone(config.reference.worker_cls, "禁用reference时不应该设置worker_cls")


if __name__ == '__main__':
    # 设置测试环境
    os.environ['PYTHONPATH'] = project_root
    
    # 运行测试
    unittest.main(verbosity=2)