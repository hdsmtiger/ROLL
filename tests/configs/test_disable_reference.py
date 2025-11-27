#!/usr/bin/env python3
"""
Test script to verify the disable_reference setting functionality.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..'))

from dataclasses import dataclass, field
from roll.pipeline.rlvr.rlvr_config import RLVRConfig
from roll.pipeline.dpo.dpo_config import DPOConfig
from roll.configs.base_config import PPOConfig

def test_rlv_rconfig():
    """Test RLVRConfig with disable_reference setting."""
    print("Testing RLVRConfig...")
    
    # Test default value
    config1 = RLVRConfig()
    assert config1.disable_reference == False, f"Expected False, got {config1.disable_reference}"
    print("✓ Default disable_reference value is False")
    
    # Test setting disable_reference to True
    config2 = RLVRConfig(disable_reference=True)
    assert config2.disable_reference == True, f"Expected True, got {config2.disable_reference}"
    print("✓ Can set disable_reference to True")
    
    # Test with YAML-like dict
    config_dict = {
        'exp_name': 'test',
        'pretrain': 'test_model',
        'disable_reference': True,
        'rollout_batch_size': 128,
        'prompt_length': 1024,
        'response_length': 1024,
        'rewards': {
            'test': {
                'worker_cls': 'test_worker',
                'model_args': {'model_name_or_path': 'test_model'},
                'data_args': {'template': 'test'},
                'world_size': 1,
                'infer_batch_size': 1
            }
        }
    }
    config3 = RLVRConfig(**config_dict)
    assert config3.disable_reference == True, f"Expected True, got {config3.disable_reference}"
    print("✓ Can initialize from dict with disable_reference=True")
    
    print("RLVRConfig tests passed!\n")

def test_dpo_config():
    """Test DPOConfig with disable_reference setting."""
    print("Testing DPOConfig...")
    
    # Test default value
    config1 = DPOConfig()
    assert config1.disable_reference == False, f"Expected False, got {config1.disable_reference}"
    print("✓ Default disable_reference value is False")
    
    # Test setting disable_reference to True
    config2 = DPOConfig(disable_reference=True)
    assert config2.disable_reference == True, f"Expected True, got {config2.disable_reference}"
    print("✓ Can set disable_reference to True")
    
    print("DPOConfig tests passed!\n")

def test_ppo_config():
    """Test PPOConfig with disable_reference setting."""
    print("Testing PPOConfig...")
    
    # Test default value
    config1 = PPOConfig()
    assert config1.disable_reference == False, f"Expected False, got {config1.disable_reference}"
    print("✓ Default disable_reference value is False")
    
    # Test setting disable_reference to True
    config2 = PPOConfig(disable_reference=True)
    assert config2.disable_reference == True, f"Expected True, got {config2.disable_reference}"
    print("✓ Can set disable_reference to True")
    
    print("PPOConfig tests passed!\n")

def main():
    """Run all tests."""
    print("Running disable_reference setting tests...\n")
    
    try:
        test_ppo_config()
        test_dpo_config()
        test_rlv_rconfig()
        
        print("🎉 All tests passed! The disable_reference setting has been successfully added.")
        print("\nUsage:")
        print("  In your configuration file, simply add:")
        print("    disable_reference: true")
        print("  to disable the reference model and save memory/computation.")
        
    except AssertionError as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()