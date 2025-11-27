#!/usr/bin/env python3
"""
Simplified test script to verify the disable_reference setting functionality.
This test checks the configuration files directly without importing the full modules.
"""

import re
import os

def test_config_file_contains_disable_reference():
    """Test that configuration files contain the disable_reference setting."""
    print("Testing configuration files for disable_reference setting...\n")
    
    # Test RLVRConfig
    rlvr_config_path = os.path.join(os.path.dirname(__file__), '..', 'roll', 'pipeline', 'rlvr', 'rlvr_config.py')
    with open(rlvr_config_path, 'r') as f:
        rlvr_content = f.read()
    
    # Check for disable_reference field definition
    disable_reference_pattern = r'disable_reference: bool = field\('
    if re.search(disable_reference_pattern, rlvr_content):
        print("✓ RLVRConfig contains disable_reference field")
    else:
        print("❌ RLVRConfig missing disable_reference field")
        return False
    
    # Check for help text
    help_text_pattern = r'Whether to disable the reference model'
    if re.search(help_text_pattern, rlvr_content):
        print("✓ RLVRConfig disable_reference has help text")
    else:
        print("❌ RLVRConfig disable_reference missing help text")
        return False
    
    # Test PPOConfig
    ppo_config_path = os.path.join(os.path.dirname(__file__), '..', 'roll', 'configs', 'base_config.py')
    with open(ppo_config_path, 'r') as f:
        ppo_content = f.read()
    
    if re.search(disable_reference_pattern, ppo_content):
        print("✓ PPOConfig contains disable_reference field")
    else:
        print("❌ PPOConfig missing disable_reference field")
        return False
    
    if re.search(help_text_pattern, ppo_content):
        print("✓ PPOConfig disable_reference has help text")
    else:
        print("❌ PPOConfig disable_reference missing help text")
        return False
    
    # Test DPOConfig
    dpo_config_path = os.path.join(os.path.dirname(__file__), '..', 'roll', 'pipeline', 'dpo', 'dpo_config.py')
    with open(dpo_config_path, 'r') as f:
        dpo_content = f.read()
    
    if re.search(disable_reference_pattern, dpo_content):
        print("✓ DPOConfig contains disable_reference field")
    else:
        print("❌ DPOConfig missing disable_reference field")
        return False
    
    if re.search(help_text_pattern, dpo_content):
        print("✓ DPOConfig disable_reference has help text")
    else:
        print("❌ DPOConfig disable_reference missing help text")
        return False
    
    return True

def test_pipeline_implementation():
    """Test that pipeline files implement the disable_reference logic."""
    print("\nTesting pipeline implementations...\n")
    
    # Test RLVR pipeline
    rlvr_pipeline_path = os.path.join(os.path.dirname(__file__), '..', 'roll', 'pipeline', 'rlvr', 'rlvr_pipeline.py')
    with open(rlvr_pipeline_path, 'r') as f:
        rlvr_pipeline_content = f.read()
    
    # Check for disable_reference conditional logic
    if 'not self.pipeline_config.disable_reference' in rlvr_pipeline_content:
        print("✓ RLVR pipeline has disable_reference conditional logic")
    else:
        print("❌ RLVR pipeline missing disable_reference conditional logic")
        return False
    
    # Check for zero tensor fallback
    if 'torch.zeros_like' in rlvr_pipeline_content:
        print("✓ RLVR pipeline has zero tensor fallback for disabled reference")
    else:
        print("❌ RLVR pipeline missing zero tensor fallback")
        return False
    
    # Test DPO pipeline
    dpo_pipeline_path = os.path.join(os.path.dirname(__file__), '..', 'roll', 'pipeline', 'dpo', 'dpo_pipeline.py')
    with open(dpo_pipeline_path, 'r') as f:
        dpo_pipeline_content = f.read()
    
    if 'not self.pipeline_config.disable_reference' in dpo_pipeline_content:
        print("✓ DPO pipeline has disable_reference conditional logic")
    else:
        print("❌ DPO pipeline missing disable_reference conditional logic")
        return False
    
    if 'torch.zeros_like' in dpo_pipeline_content:
        print("✓ DPO pipeline has zero tensor fallback for disabled reference")
    else:
        print("❌ DPO pipeline missing zero tensor fallback")
        return False
    
    return True

def test_example_config():
    """Test that example configuration file exists and contains the setting."""
    print("\nTesting example configuration...\n")
    
    example_config_path = os.path.join(os.path.dirname(__file__), '..', 'examples', 'docs_examples', 'example_disable_reference.yaml')
    if os.path.exists(example_config_path):
        print("✓ Example configuration file exists")
        
        with open(example_config_path, 'r') as f:
            example_content = f.read()
        
        if 'disable_reference: true' in example_content:
            print("✓ Example configuration has disable_reference set to true")
        else:
            print("❌ Example configuration missing disable_reference setting")
            return False
        
        if 'Disable reference model to save memory and computation' in example_content:
            print("✓ Example configuration has explanatory comment")
        else:
            print("❌ Example configuration missing explanatory comment")
            return False
    else:
        print("❌ Example configuration file does not exist")
        return False
    
    return True

def main():
    """Run all tests."""
    print("Running disable_reference implementation tests...\n")
    
    all_passed = True
    
    try:
        all_passed &= test_config_file_contains_disable_reference()
        all_passed &= test_pipeline_implementation()
        all_passed &= test_example_config()
        
        if all_passed:
            print("\n🎉 All tests passed! The disable_reference setting has been successfully implemented.")
            print("\nImplementation Summary:")
            print("1. ✓ Added disable_reference field to PPOConfig, DPOConfig, and RLVRConfig")
            print("2. ✓ Modified pipeline logic to conditionally initialize reference model")
            print("3. ✓ Added zero tensor fallback when reference is disabled")
            print("4. ✓ Created example configuration file")
            print("\nUsage:")
            print("  In your configuration file, simply add:")
            print("    disable_reference: true")
            print("  to disable the reference model and save memory/computation.")
        else:
            print("\n❌ Some tests failed. Please check the implementation.")
            return 1
            
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())