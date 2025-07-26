"""Test configuration schemas and Hydra config loading.

This module validates that all Hydra configuration files can be loaded
correctly and contain the expected keys and structure.
"""

import pytest
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
import os
import sys

from src.ball.tests.fixtures.mock_data import get_mock_config_paths


class TestConfigSchemas:
    """Test suite for validating Hydra configuration schemas."""
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        # Clear Hydra state before each test
        GlobalHydra.instance().clear()
        yield
        # Clear Hydra state after each test
        GlobalHydra.instance().clear()
    
    def test_config_files_exist(self):
        """Test that all required config files exist."""
        config_paths = get_mock_config_paths()
        
        for config_name, config_path in config_paths.items():
            path = Path(config_path)
            assert path.exists(), f"Config file {config_name} not found at {config_path}"
            assert path.suffix in ['.yaml', '.yml'], f"Config file {config_name} should be YAML"
    
    def test_config_files_are_valid_yaml(self):
        """Test that all config files are valid YAML."""
        config_paths = get_mock_config_paths()
        
        for config_name, config_path in config_paths.items():
            try:
                config = OmegaConf.load(config_path)
                assert isinstance(config, DictConfig), f"Config {config_name} should be DictConfig"
            except Exception as e:
                pytest.fail(f"Failed to load {config_name} as YAML: {e}")
    
    def test_litmodule_config_schema(self):
        """Test LitModule config has required keys."""
        config_paths = get_mock_config_paths()
        config = OmegaConf.load(config_paths["litmodule_config"])
        
        # Required top-level keys
        required_keys = {"_target_", "model", "criterion"}
        missing_keys = required_keys - set(config.keys())
        assert not missing_keys, f"LitModule config missing keys: {missing_keys}"
        
        # Check _target_ points to correct class
        assert "LitBallModel" in config["_target_"], "LitModule _target_ should reference LitBallModel"
        
        # Check criterion config
        criterion = config["criterion"]
        assert "alpha" in criterion, "Criterion missing alpha parameter"
        assert "gamma" in criterion, "Criterion missing gamma parameter"
        assert isinstance(criterion["alpha"], (int, float)), "Alpha should be numeric"
        assert isinstance(criterion["gamma"], (int, float)), "Gamma should be numeric"
        assert criterion["alpha"] > 0, "Alpha should be positive"
        assert criterion["gamma"] >= 0, "Gamma should be non-negative"
    
    def test_model_config_schema(self):
        """Test model config has required keys."""
        config_paths = get_mock_config_paths()
        config = OmegaConf.load(config_paths["model_config"])
        
        # Required top-level keys for model config
        required_keys = {"_target_"}
        missing_keys = required_keys - set(config.keys())
        assert not missing_keys, f"Model config missing keys: {missing_keys}"
        
        # Check _target_ points to correct model class
        target = config["_target_"]
        assert "conv3d_tsm_fpn" in target.lower() or "Conv3dTsmFpn" in target, \
            f"Model _target_ should reference Conv3D TSM FPN model, got: {target}"
    
    def test_datamodule_config_schema(self):
        """Test DataModule config has required keys.""" 
        config_paths = get_mock_config_paths()
        datamodule_config_path = config_paths["datamodule_config"]
        
        if Path(datamodule_config_path).exists():
            config = OmegaConf.load(datamodule_config_path)
            
            # Check common datamodule keys
            if "_target_" in config:
                assert "BallDataModule" in config["_target_"] or "DataModule" in config["_target_"], \
                    "DataModule _target_ should reference BallDataModule"
    
    def test_main_config_composition(self):
        """Test main config can be composed with Hydra."""
        # Get the config directory - adapt for Docker or local environment
        if Path("/workspace").exists():
            config_dir = Path("/workspace/configs/train/ball").resolve()
        else:
            config_dir = Path("/mnt/c/Users/kamim/code/tennis_systems/configs/train/ball").resolve()
        assert config_dir.exists(), f"Config directory not found: {config_dir}"
        
        try:
            with initialize_config_dir(config_dir=str(config_dir), version_base=None):
                # Test composing the main config
                cfg = compose(config_name="conv3d_tsm_fpn.yaml")
                
                # Check that composition succeeded
                assert cfg is not None, "Failed to compose config"
                assert isinstance(cfg, DictConfig), "Composed config should be DictConfig"
                
                # Check for expected top-level structure
                present_sections = set(cfg.keys())
                
                # Log what we found for debugging
                print(f"Config sections found: {present_sections}")
                
                # Check for essential config sections (after composition, defaults are resolved)
                essential_sections = {"model", "litmodule"}
                missing_sections = essential_sections - present_sections
                assert not missing_sections, f"Config missing essential sections: {missing_sections}"
                
        except Exception as e:
            pytest.fail(f"Failed to compose main config: {e}")
    
    def test_config_parameter_types(self):
        """Test that config parameters have correct types."""
        config_paths = get_mock_config_paths()
        
        # Test LitModule config types
        litmodule_config = OmegaConf.load(config_paths["litmodule_config"])
        
        if "criterion" in litmodule_config:
            criterion = litmodule_config["criterion"]
            
            # Test alpha parameter
            if "alpha" in criterion:
                alpha = criterion["alpha"]
                assert isinstance(alpha, (int, float)), f"Alpha should be numeric, got {type(alpha)}"
                assert 0 < alpha <= 1, f"Alpha should be in (0,1], got {alpha}"
            
            # Test gamma parameter  
            if "gamma" in criterion:
                gamma = criterion["gamma"]
                assert isinstance(gamma, (int, float)), f"Gamma should be numeric, got {type(gamma)}"
                assert gamma >= 0, f"Gamma should be non-negative, got {gamma}"
        
        # Test logging config if present
        if "logging_config" in litmodule_config:
            logging_config = litmodule_config["logging_config"]
            
            if "num_log_images" in logging_config:
                num_log_images = logging_config["num_log_images"]
                assert isinstance(num_log_images, int), "num_log_images should be integer"
                assert num_log_images >= 0, "num_log_images should be non-negative"
    
    def test_config_interpolations(self):
        """Test that config interpolations resolve correctly."""
        config_paths = get_mock_config_paths()
        litmodule_config = OmegaConf.load(config_paths["litmodule_config"])
        
        # Check for interpolation syntax
        config_str = OmegaConf.to_yaml(litmodule_config)
        
        # Look for common interpolation patterns
        interpolations = []
        for line in config_str.split('\n'):
            if '${' in line:
                interpolations.append(line.strip())
        
        print(f"Found interpolations: {interpolations}")
        
        # Test that interpolations use valid syntax
        for interp in interpolations:
            # Should contain ${key} format
            assert '${' in interp and '}' in interp, f"Invalid interpolation syntax: {interp}"
    
    def test_config_consistency_across_files(self):
        """Test consistency of related configurations."""
        config_paths = get_mock_config_paths()
        
        # Load multiple configs
        configs = {}
        for name, path in config_paths.items():
            if Path(path).exists():
                configs[name] = OmegaConf.load(path)
        
        # Test that model references are consistent
        if "litmodule_config" in configs and "model_config" in configs:
            litmodule = configs["litmodule_config"]
            model = configs["model_config"]
            
            # Both should reference compatible targets
            if "_target_" in litmodule and "_target_" in model:
                lit_target = litmodule["_target_"]
                model_target = model["_target_"]
                
                # Basic sanity check - both should be valid Python module paths
                assert "." in lit_target, "LitModule target should be module path"
                assert "." in model_target, "Model target should be module path"
    
    def test_config_environment_variables(self):
        """Test handling of environment variables in configs."""
        # Test with mock environment variables
        test_env_vars = {
            "BALL_MODEL_PATH": "/tmp/test/model",
            "BALL_DATA_PATH": "/tmp/test/data"
        }
        
        original_env = {}
        try:
            # Set test environment variables
            for key, value in test_env_vars.items():
                original_env[key] = os.environ.get(key)
                os.environ[key] = value
            
            # Test config loading with environment variables
            config_paths = get_mock_config_paths()
            for config_name, config_path in config_paths.items():
                if Path(config_path).exists():
                    try:
                        config = OmegaConf.load(config_path)
                        # Config should load without errors even with env vars
                        assert config is not None
                    except Exception as e:
                        pytest.fail(f"Config {config_name} failed with env vars: {e}")
        
        finally:
            # Restore original environment
            for key, original_value in original_env.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value
    
    def test_config_defaults_resolution(self):
        """Test that default configurations resolve properly."""
        # Adapt for Docker or local environment
        if Path("/workspace").exists():
            config_dir = Path("/workspace/configs/train/ball").resolve()
        else:
            config_dir = Path("/mnt/c/Users/kamim/code/tennis_systems/configs/train/ball").resolve()
        
        if not config_dir.exists():
            pytest.skip(f"Config directory not found: {config_dir}")
            
        try:
            with initialize_config_dir(config_dir=str(config_dir), version_base=None):
                cfg = compose(config_name="conv3d_tsm_fpn.yaml")
                
                # Check that defaults were applied
                if "defaults" in cfg:
                    defaults = cfg["defaults"] 
                    print(f"Config defaults: {defaults}")
                    
                    # Defaults should be a list
                    assert isinstance(defaults, (list, DictConfig)), \
                        f"Defaults should be list or DictConfig, got {type(defaults)}"
                
                # Test that resolved config has expected structure
                # The exact structure depends on the defaults composition
                print(f"Resolved config keys: {list(cfg.keys())}")
                
                # Check for essential sections in resolved config
                essential_keys = {"model", "litmodule"}
                present_keys = set(cfg.keys())
                missing_keys = essential_keys - present_keys
                assert not missing_keys, f"Resolved config missing essential keys: {missing_keys}"
                
        except Exception as e:
            pytest.fail(f"Failed to resolve config defaults: {e}")
    
    def test_config_seed_parameter(self):
        """Test that seed parameter is properly configured."""
        config_paths = get_mock_config_paths()
        main_config_path = config_paths["main_config"]
        
        if Path(main_config_path).exists():
            config = OmegaConf.load(main_config_path)
            
            if "seed" in config:
                seed = config["seed"]
                assert isinstance(seed, int), f"Seed should be integer, got {type(seed)}"
                assert seed >= 0, f"Seed should be non-negative, got {seed}"
                assert seed <= 2**32 - 1, f"Seed too large: {seed}"