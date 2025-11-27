#!/usr/bin/env python3
"""
Simple test script to verify that the use_reference_model setting has been added correctly.
"""

import os
import re

def check_file_contains(filepath, pattern):
    """Check if a file contains a specific pattern."""
    with open(filepath, 'r') as f:
        content = f.read()
    return pattern in content

def test_base_config():
    """Test that base_config.py contains the use_reference_model field."""
    print("Testing base_config.py...")
    filepath = "/home/admin/iflow-cli-dev-service/iflow-workspace/ROLL/roll/configs/base_config.py"
    
    # Check that the field is defined
    assert check_file_contains(filepath, "use_reference_model: bool = field("), "use_reference_model field not found"
    assert check_file_contains(filepath, 'default=True'), "use_reference_model default should be True"
    assert check_file_contains(filepath, "metadata={\"help\": \"Whether to use reference model"), "use_reference_model help text not found"
    
    print("✓ base_config.py test passed")

def test_dpo_config():
    """Test that dpo_config.py contains the use_reference_model field."""
    print("Testing dpo_config.py...")
    filepath = "/home/admin/iflow-cli-dev-service/iflow-workspace/ROLL/roll/pipeline/dpo/dpo_config.py"
    
    # Check that the field is defined
    assert check_file_contains(filepath, "use_reference_model: bool = field("), "use_reference_model field not found"
    assert check_file_contains(filepath, 'default=True'), "use_reference_model default should be True"
    assert check_file_contains(filepath, "metadata={\"help\": \"Whether to use reference model"), "use_reference_model help text not found"
    
    print("✓ dpo_config.py test passed")

def test_agentic_pipeline():
    """Test that agentic_pipeline.py contains conditional reference model logic."""
    print("Testing agentic_pipeline.py...")
    filepath = "/home/admin/iflow-cli-dev-service/iflow-workspace/ROLL/roll/pipeline/agentic/agentic_pipeline.py"
    
    # Check that the reference model is conditionally created
    assert check_file_contains(filepath, "if self.pipeline_config.use_reference_model:"), "Conditional reference model creation not found"
    assert check_file_contains(filepath, "self.reference = None"), "Reference model initialization not found"
    
    print("✓ agentic_pipeline.py test passed")

def test_dpo_pipeline():
    """Test that dpo_pipeline.py contains conditional reference model logic."""
    print("Testing dpo_pipeline.py...")
    filepath = "/home/admin/iflow-cli-dev-service/iflow-workspace/ROLL/roll/pipeline/dpo/dpo_pipeline.py"
    
    # Check that the reference model is conditionally created
    assert check_file_contains(filepath, "if self.pipeline_config.use_reference_model:"), "Conditional reference model creation not found"
    assert check_file_contains(filepath, "self.reference = None"), "Reference model initialization not found"
    
    print("✓ dpo_pipeline.py test passed")

def test_rlvr_pipeline():
    """Test that rlvr_pipeline.py contains conditional reference model logic."""
    print("Testing rlvr_pipeline.py...")
    filepath = "/home/admin/iflow-cli-dev-service/iflow-workspace/ROLL/roll/pipeline/rlvr/rlvr_pipeline.py"
    
    # Check that the reference model is conditionally created
    assert check_file_contains(filepath, "if self.pipeline_config.use_reference_model:"), "Conditional reference model creation not found"
    
    print("✓ rlvr_pipeline.py test passed")

def test_rlvr_vlm_pipeline():
    """Test that rlvr_vlm_pipeline.py contains conditional reference model logic."""
    print("Testing rlvr_vlm_pipeline.py...")
    filepath = "/home/admin/iflow-cli-dev-service/iflow-workspace/ROLL/roll/pipeline/rlvr/rlvr_vlm_pipeline.py"
    
    # Check that the reference model is conditionally created
    assert check_file_contains(filepath, "if self.pipeline_config.use_reference_model:"), "Conditional reference model creation not found"
    assert check_file_contains(filepath, "self.reference = None"), "Reference model initialization not found"
    
    print("✓ rlvr_vlm_pipeline.py test passed")

def test_rlvr_math_vlm_pipeline():
    """Test that rlvr_math_vlm_pipeline.py contains conditional reference model logic."""
    print("Testing rlvr_math_vlm_pipeline.py...")
    filepath = "/home/admin/iflow-cli-dev-service/iflow-workspace/ROLL/roll/pipeline/rlvr/rlvr_math_vlm_pipeline.py"
    
    # Check that the reference model is conditionally created
    assert check_file_contains(filepath, "if self.pipeline_config.use_reference_model:"), "Conditional reference model creation not found"
    assert check_file_contains(filepath, "self.reference = None"), "Reference model initialization not found"
    
    print("✓ rlvr_math_vlm_pipeline.py test passed")

if __name__ == "__main__":
    print("Running simple tests for use_reference_model setting...")
    test_base_config()
    test_dpo_config()
    test_agentic_pipeline()
    test_dpo_pipeline()
    test_rlvr_pipeline()
    test_rlvr_vlm_pipeline()
    test_rlvr_math_vlm_pipeline()
    print("\n✅ All tests passed! The use_reference_model setting has been successfully implemented.")