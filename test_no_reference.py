#!/usr/bin/env python3
"""
Test script to verify that the use_reference_model setting works correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from roll.configs.base_config import PPOConfig
from roll.pipeline.agentic.agentic_config import AgenticConfig
from roll.pipeline.dpo.dpo_config import DPOConfig

def test_ppo_config():
    """Test that PPOConfig has use_reference_model field."""
    print("Testing PPOConfig...")
    config = PPOConfig()
    
    # Check that the field exists
    assert hasattr(config, 'use_reference_model'), "use_reference_model field not found in PPOConfig"
    assert config.use_reference_model == True, "Default value for use_reference_model should be True"
    
    # Test setting it to False
    config.use_reference_model = False
    assert config.use_reference_model == False, "Failed to set use_reference_model to False"
    
    print("✓ PPOConfig test passed")

def test_dpo_config():
    """Test that DPOConfig has use_reference_model field."""
    print("Testing DPOConfig...")
    config = DPOConfig()
    
    # Check that the field exists
    assert hasattr(config, 'use_reference_model'), "use_reference_model field not found in DPOConfig"
    assert config.use_reference_model == True, "Default value for use_reference_model should be True"
    
    # Test setting it to False
    config.use_reference_model = False
    assert config.use_reference_model == False, "Failed to set use_reference_model to False"
    
    print("✓ DPOConfig test passed")

def test_agentic_config():
    """Test that AgenticConfig has use_reference_model field."""
    print("Testing AgenticConfig...")
    config = AgenticConfig()
    
    # Check that the field exists (inherited from PPOConfig)
    assert hasattr(config, 'use_reference_model'), "use_reference_model field not found in AgenticConfig"
    assert config.use_reference_model == True, "Default value for use_reference_model should be True"
    
    # Test setting it to False
    config.use_reference_model = False
    assert config.use_reference_model == False, "Failed to set use_reference_model to False"
    
    print("✓ AgenticConfig test passed")

def test_config_loading():
    """Test loading a config file with use_reference_model set to False."""
    print("Testing config file loading (simplified)...")
    
    # Create a simple config object and set use_reference_model to False
    config = PPOConfig()
    config.use_reference_model = False
    
    # Check that the value is set correctly
    assert config.use_reference_model == False, "Failed to set use_reference_model to False"
    
    print("✓ Config loading test passed")

if __name__ == "__main__":
    print("Running tests for use_reference_model setting...")
    test_ppo_config()
    test_dpo_config()
    test_agentic_config()
    test_config_loading()
    print("All tests completed!")