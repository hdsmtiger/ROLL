#!/usr/bin/env python3
"""
简化的单元测试：验证disable_reference功能的核心逻辑
"""

import os
import sys
import unittest

class TestDisableReferenceBasic(unittest.TestCase):
    """测试disable_reference基本功能"""
    
    def setUp(self):
        """测试前的准备工作"""
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    def test_config_files_contain_disable_reference(self):
        """测试配置文件是否包含disable_reference字段"""
        
        # 检查base_config.py
        base_config_path = os.path.join(self.project_root, "roll/configs/base_config.py")
        with open(base_config_path, 'r') as f:
            base_config_content = f.read()
        
        self.assertIn("disable_reference: bool", base_config_content, 
                      "base_config.py应该包含disable_reference字段")
        self.assertIn("not self.disable_reference", base_config_content,
                      "base_config.py应该有disable_reference的条件判断")
        
        # 检查dpo_config.py
        dpo_config_path = os.path.join(self.project_root, "roll/pipeline/dpo/dpo_config.py")
        with open(dpo_config_path, 'r') as f:
            dpo_config_content = f.read()
        
        self.assertIn("disable_reference: bool", dpo_config_content,
                      "dpo_config.py应该包含disable_reference字段")
        self.assertIn("not self.disable_reference", dpo_config_content,
                      "dpo_config.py应该有disable_reference的条件判断")
        
        # 检查agentic_config.py
        agentic_config_path = os.path.join(self.project_root, "roll/pipeline/agentic/agentic_config.py")
        with open(agentic_config_path, 'r') as f:
            agentic_config_content = f.read()
        
        self.assertIn("not self.disable_reference and self.reference.worker_cls is None", 
                      agentic_config_content,
                      "agentic_config.py应该有disable_reference的条件判断")
        
        # 检查rlvr_config.py
        rlvr_config_path = os.path.join(self.project_root, "roll/pipeline/rlvr/rlvr_config.py")
        with open(rlvr_config_path, 'r') as f:
            rlvr_config_content = f.read()
        
        self.assertIn("not self.disable_reference and self.reference.worker_cls is None",
                      rlvr_config_content,
                      "rlvr_config.py应该有disable_reference的条件判断")
    
    def test_pipeline_files_handle_disable_reference(self):
        """测试pipeline文件是否正确处理disable_reference"""
        
        # 检查dpo_pipeline.py
        dpo_pipeline_path = os.path.join(self.project_root, "roll/pipeline/dpo/dpo_pipeline.py")
        with open(dpo_pipeline_path, 'r') as f:
            dpo_pipeline_content = f.read()
        
        self.assertIn("not self.pipeline_config.disable_reference", dpo_pipeline_content,
                      "dpo_pipeline.py应该有disable_reference的条件判断")
        self.assertIn("When reference is disabled", dpo_pipeline_content,
                      "dpo_pipeline.py应该有禁用reference时的处理逻辑")
        
        # 检查agentic_pipeline.py
        agentic_pipeline_path = os.path.join(self.project_root, "roll/pipeline/agentic/agentic_pipeline.py")
        with open(agentic_pipeline_path, 'r') as f:
            agentic_pipeline_content = f.read()
        
        self.assertIn("not self.pipeline_config.disable_reference", agentic_pipeline_content,
                      "agentic_pipeline.py应该有disable_reference的条件判断")
        self.assertIn("When reference is disabled", agentic_pipeline_content,
                      "agentic_pipeline.py应该有禁用reference时的处理逻辑")
        
        # 检查rlvr_pipeline.py
        rlvr_pipeline_path = os.path.join(self.project_root, "roll/pipeline/rlvr/rlvr_pipeline.py")
        with open(rlvr_pipeline_path, 'r') as f:
            rlvr_pipeline_content = f.read()
        
        self.assertIn("not self.pipeline_config.disable_reference", rlvr_pipeline_content,
                      "rlvr_pipeline.py应该有disable_reference的条件判断")
        self.assertIn("When reference is disabled", rlvr_pipeline_content,
                      "rlvr_pipeline.py应该有禁用reference时的处理逻辑")
    
    def test_disable_reference_logic_consistency(self):
        """测试disable_reference逻辑的一致性"""
        
        # 检查所有文件中的disable_reference使用是否一致
        files_to_check = [
            "roll/configs/base_config.py",
            "roll/pipeline/dpo/dpo_config.py",
            "roll/pipeline/agentic/agentic_config.py",
            "roll/pipeline/rlvr/rlvr_config.py",
            "roll/pipeline/dpo/dpo_pipeline.py",
            "roll/pipeline/agentic/agentic_pipeline.py",
            "roll/pipeline/rlvr/rlvr_pipeline.py"
        ]
        
        for file_path in files_to_check:
            full_path = os.path.join(self.project_root, file_path)
            with open(full_path, 'r') as f:
                content = f.read()
            
            # 检查是否有disable_reference字段的定义
            if "Config" in file_path and "pipeline" not in file_path:
                self.assertIn("disable_reference: bool", content,
                              f"{file_path}应该包含disable_reference字段定义")
            
            # 检查是否有disable_reference的条件判断
            if "pipeline" in file_path:
                self.assertTrue(
                    "not self.pipeline_config.disable_reference" in content or 
                    "not self.disable_reference" in content,
                    f"{file_path}应该有disable_reference的条件判断"
                )
    
    def test_backward_compatibility(self):
        """测试向后兼容性"""
        
        # 检查默认值设置
        base_config_path = os.path.join(self.project_root, "roll/configs/base_config.py")
        with open(base_config_path, 'r') as f:
            base_config_content = f.read()
        
        self.assertIn("default=False", base_config_content,
                      "disable_reference的默认值应该为False，确保向后兼容")
    
    def test_fallback_logic(self):
        """测试回退逻辑"""
        
        # 检查dpo_pipeline.py中的回退逻辑
        dpo_pipeline_path = os.path.join(self.project_root, "roll/pipeline/dpo/dpo_pipeline.py")
        with open(dpo_pipeline_path, 'r') as f:
            dpo_pipeline_content = f.read()
        
        # 检查是否有使用actor log probs作为reference的逻辑
        self.assertTrue(
            "actor_log_probs = self.actor_train.compute_log_probs" in dpo_pipeline_content or
            "actor_infer.compute_log_probs" in dpo_pipeline_content,
            "应该有使用actor log probs作为reference的回退逻辑"
        )
        
        # 检查agentic_pipeline.py中的回退逻辑
        agentic_pipeline_path = os.path.join(self.project_root, "roll/pipeline/agentic/agentic_pipeline.py")
        with open(agentic_pipeline_path, 'r') as f:
            agentic_pipeline_content = f.read()
        
        # 检查是否有使用actor log probs作为reference的逻辑
        self.assertTrue(
            "self.actor_train.compute_log_probs" in agentic_pipeline_content,
            "应该有使用actor log probs作为reference的回退逻辑"
        )
        
        # 检查rlvr_pipeline.py中的回退逻辑
        rlvr_pipeline_path = os.path.join(self.project_root, "roll/pipeline/rlvr/rlvr_pipeline.py")
        with open(rlvr_pipeline_path, 'r') as f:
            rlvr_pipeline_content = f.read()
        
        # 检查是否有使用actor log probs作为reference的逻辑
        self.assertTrue(
            "self.actor_train.compute_log_probs" in rlvr_pipeline_content,
            "应该有使用actor log probs作为reference的回退逻辑"
        )


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)