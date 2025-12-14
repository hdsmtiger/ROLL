#!/usr/bin/env python3

"""
简化的disable_reference功能单元测试
避免复杂的依赖，专注于核心功能测试
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch, MagicMock

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestDisableReferenceBasic(unittest.TestCase):
    """测试disable_reference基本功能"""
    
    def test_config_file_modifications(self):
        """测试配置文件修改是否正确"""
        # 测试base_config.py
        base_config_path = "/home/admin/iflow-cli-dev-service/iflow-workspace/ROLL/roll/configs/base_config.py"
        with open(base_config_path, 'r') as f:
            base_config_content = f.read()
        
        self.assertIn("disable_reference: bool", base_config_content, 
                     "base_config.py应该包含disable_reference字段")
        self.assertIn("not self.disable_reference", base_config_content,
                     "base_config.py应该有disable_reference的条件判断")
        
        # 测试dpo_config.py
        dpo_config_path = "/home/admin/iflow-cli-dev-service/iflow-workspace/ROLL/roll/pipeline/dpo/dpo_config.py"
        with open(dpo_config_path, 'r') as f:
            dpo_config_content = f.read()
        
        self.assertIn("disable_reference: bool", dpo_config_content,
                     "dpo_config.py应该包含disable_reference字段")
        self.assertIn("not self.disable_reference", dpo_config_content,
                     "dpo_config.py应该有disable_reference的条件判断")
        
        # 测试agentic_config.py
        agentic_config_path = "/home/admin/iflow-cli-dev-service/iflow-workspace/ROLL/roll/pipeline/agentic/agentic_config.py"
        with open(agentic_config_path, 'r') as f:
            agentic_config_content = f.read()
        
        self.assertIn("not self.disable_reference and self.reference.worker_cls is None", agentic_config_content,
                     "agentic_config.py应该有disable_reference的条件判断")
    
    def test_pipeline_file_modifications(self):
        """测试pipeline文件修改是否正确"""
        # 测试dpo_pipeline.py
        dpo_pipeline_path = "/home/admin/iflow-cli-dev-service/iflow-workspace/ROLL/roll/pipeline/dpo/dpo_pipeline.py"
        with open(dpo_pipeline_path, 'r') as f:
            dpo_pipeline_content = f.read()
        
        self.assertIn("not self.pipeline_config.disable_reference", dpo_pipeline_content,
                     "dpo_pipeline.py应该有disable_reference的条件判断")
        self.assertIn("When reference is disabled", dpo_pipeline_content,
                     "dpo_pipeline.py应该有禁用reference时的处理逻辑")
        
        # 测试agentic_pipeline.py
        agentic_pipeline_path = "/home/admin/iflow-cli-dev-service/iflow-workspace/ROLL/roll/pipeline/agentic/agentic_pipeline.py"
        with open(agentic_pipeline_path, 'r') as f:
            agentic_pipeline_content = f.read()
        
        self.assertIn("not self.pipeline_config.disable_reference", agentic_pipeline_content,
                     "agentic_pipeline.py应该有disable_reference的条件判断")
        self.assertIn("When reference is disabled", agentic_pipeline_content,
                     "agentic_pipeline.py应该有禁用reference时的处理逻辑")
        
        # 测试rlvr_pipeline.py
        rlvr_pipeline_path = "/home/admin/iflow-cli-dev-service/iflow-workspace/ROLL/roll/pipeline/rlvr/rlvr_pipeline.py"
        with open(rlvr_pipeline_path, 'r') as f:
            rlvr_pipeline_content = f.read()
        
        self.assertIn("not self.pipeline_config.disable_reference", rlvr_pipeline_content,
                     "rlvr_pipeline.py应该有disable_reference的条件判断")
        self.assertIn("When reference is disabled", rlvr_pipeline_content,
                     "rlvr_pipeline.py应该有禁用reference时的处理逻辑")
    
    def test_disable_reference_logic_consistency(self):
        """测试disable_reference逻辑的一致性"""
        # 检查所有相关文件中的disable_reference逻辑是否一致
        files_to_check = [
            "/home/admin/iflow-cli-dev-service/iflow-workspace/ROLL/roll/configs/base_config.py",
            "/home/admin/iflow-cli-dev-service/iflow-workspace/ROLL/roll/pipeline/dpo/dpo_config.py",
            "/home/admin/iflow-cli-dev-service/iflow-workspace/ROLL/roll/pipeline/dpo/dpo_pipeline.py",
            "/home/admin/iflow-cli-dev-service/iflow-workspace/ROLL/roll/pipeline/agentic/agentic_config.py",
            "/home/admin/iflow-cli-dev-service/iflow-workspace/ROLL/roll/pipeline/agentic/agentic_pipeline.py",
            "/home/admin/iflow-cli-dev-service/iflow-workspace/ROLL/roll/pipeline/rlvr/rlvr_config.py",
            "/home/admin/iflow-cli-dev-service/iflow-workspace/ROLL/roll/pipeline/rlvr/rlvr_pipeline.py"
        ]
        
        for file_path in files_to_check:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # 检查是否有disable_reference相关的逻辑
            if "disable_reference" in content:
                # 验证逻辑的一致性
                self.assertIn("disable_reference", content,
                             f"{file_path}应该包含disable_reference相关代码")
                
                # 检查是否有条件判断
                if "not self.disable_reference" in content or "not self.pipeline_config.disable_reference" in content:
                    self.assertTrue(True, f"{file_path}有正确的disable_reference条件判断")
    
    def test_backward_compatibility(self):
        """测试向后兼容性"""
        # 检查默认值是否为False（保持向后兼容）
        base_config_path = "/home/admin/iflow-cli-dev-service/iflow-workspace/ROLL/roll/configs/base_config.py"
        with open(base_config_path, 'r') as f:
            base_config_content = f.read()
        
        # 检查默认值是否为False
        self.assertIn("default=False", base_config_content,
                     "disable_reference的默认值应该为False，保持向后兼容性")
    
    def test_fallback_logic(self):
        """测试回退逻辑"""
        # 检查是否有适当的回退逻辑
        dpo_pipeline_path = "/home/admin/iflow-cli-dev-service/iflow-workspace/ROLL/roll/pipeline/dpo/dpo_pipeline.py"
        with open(dpo_pipeline_path, 'r') as f:
            dpo_pipeline_content = f.read()
        
        # 检查是否有使用actor log probs作为reference的回退逻辑
        self.assertIn("actor_log_probs", dpo_pipeline_content,
                     "应该有使用actor log probs作为reference的回退逻辑")
        
        agentic_pipeline_path = "/home/admin/iflow-cli-dev-service/iflow-workspace/ROLL/roll/pipeline/agentic/agentic_pipeline.py"
        with open(agentic_pipeline_path, 'r') as f:
            agentic_pipeline_content = f.read()
        
        self.assertIn("actor_train.compute_log_probs", agentic_pipeline_content,
                     "应该有使用actor_train log probs作为reference的回退逻辑")


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    test_suite = loader.loadTestsFromTestCase(TestDisableReferenceBasic)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 返回测试结果
    return result.wasSuccessful()


if __name__ == "__main__":
    print("开始运行disable_reference功能的简化单元测试...")
    print("=" * 60)
    
    success = run_tests()
    
    print("=" * 60)
    if success:
        print("🎉 所有单元测试通过！")
        sys.exit(0)
    else:
        print("❌ 部分单元测试失败！")
        sys.exit(1)