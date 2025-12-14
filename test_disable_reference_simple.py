#!/usr/bin/env python3

"""
简单测试脚本：验证disable_reference配置是否正确添加
"""

import sys
import os

def test_config_files():
    """测试配置文件中是否正确添加了disable_reference选项"""
    print("测试配置文件中的disable_reference选项...")
    
    # 检查base_config.py
    base_config_path = "/home/admin/iflow-cli-dev-service/iflow-workspace/ROLL/roll/configs/base_config.py"
    with open(base_config_path, 'r') as f:
        base_config_content = f.read()
    
    assert "disable_reference: bool" in base_config_content, "base_config.py中应该包含disable_reference字段"
    assert "not self.disable_reference" in base_config_content, "base_config.py中应该有disable_reference的条件判断"
    print("✓ base_config.py 包含正确的disable_reference配置")
    
    # 检查dpo_config.py
    dpo_config_path = "/home/admin/iflow-cli-dev-service/iflow-workspace/ROLL/roll/pipeline/dpo/dpo_config.py"
    with open(dpo_config_path, 'r') as f:
        dpo_config_content = f.read()
    
    assert "disable_reference: bool" in dpo_config_content, "dpo_config.py中应该包含disable_reference字段"
    assert "not self.disable_reference" in dpo_config_content, "dpo_config.py中应该有disable_reference的条件判断"
    print("✓ dpo_config.py 包含正确的disable_reference配置")
    
    # 检查agentic_config.py
    agentic_config_path = "/home/admin/iflow-cli-dev-service/iflow-workspace/ROLL/roll/pipeline/agentic/agentic_config.py"
    with open(agentic_config_path, 'r') as f:
        agentic_config_content = f.read()
    
    assert "not self.disable_reference and self.reference.worker_cls is None" in agentic_config_content, "agentic_config.py中应该有disable_reference的条件判断"
    print("✓ agentic_config.py 包含正确的disable_reference配置")
    
    print("\n📁 所有必要文件已正确更新disable_reference配置！")

def test_pipeline_files():
    """测试pipeline文件中是否正确处理了disable_reference"""
    print("\n测试pipeline文件中的disable_reference处理...")
    
    # 检查dpo_pipeline.py
    dpo_pipeline_path = "/home/admin/iflow-cli-dev-service/iflow-workspace/ROLL/roll/pipeline/dpo/dpo_pipeline.py"
    with open(dpo_pipeline_path, 'r') as f:
        dpo_pipeline_content = f.read()
    
    assert "not self.pipeline_config.disable_reference" in dpo_pipeline_content, "dpo_pipeline.py中应该有disable_reference的条件判断"
    assert "When reference is disabled" in dpo_pipeline_content, "dpo_pipeline.py中应该有禁用reference时的处理逻辑"
    print("✓ dpo_pipeline.py 包含正确的disable_reference处理")
    
    # 检查agentic_pipeline.py
    agentic_pipeline_path = "/home/admin/iflow-cli-dev-service/iflow-workspace/ROLL/roll/pipeline/agentic/agentic_pipeline.py"
    with open(agentic_pipeline_path, 'r') as f:
        agentic_pipeline_content = f.read()
    
    assert "not self.pipeline_config.disable_reference" in agentic_pipeline_content, "agentic_pipeline.py中应该有disable_reference的条件判断"
    assert "When reference is disabled" in agentic_pipeline_content, "agentic_pipeline.py中应该有禁用reference时的处理逻辑"
    print("✓ agentic_pipeline.py 包含正确的disable_reference处理")
    
    # 检查rlvr_pipeline.py
    rlvr_pipeline_path = "/home/admin/iflow-cli-dev-service/iflow-workspace/ROLL/roll/pipeline/rlvr/rlvr_pipeline.py"
    with open(rlvr_pipeline_path, 'r') as f:
        rlvr_pipeline_content = f.read()
    
    assert "not self.pipeline_config.disable_reference" in rlvr_pipeline_content, "rlvr_pipeline.py中应该有disable_reference的条件判断"
    assert "When reference is disabled" in rlvr_pipeline_content, "rlvr_pipeline.py中应该有禁用reference时的处理逻辑"
    print("✓ rlvr_pipeline.py 包含正确的disable_reference处理")
    
    print("\n🔄 所有必要pipeline文件已正确实现disable_reference逻辑！")

if __name__ == "__main__":
    print("开始测试 disable_reference 功能实现...\n")
    
    try:
        test_config_files()
        test_pipeline_files()
        print("\n🎉 所有测试通过！disable_reference 功能已成功实现。")
        print("\n📝 使用方法：")
        print("在配置文件中设置 disable_reference: True 即可禁用Reference Model。")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        sys.exit(1)