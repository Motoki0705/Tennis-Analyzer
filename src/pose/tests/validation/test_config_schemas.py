"""Test configuration schemas for pose estimation."""

import pytest
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from src.pose.tests.fixtures.mock_data import get_mock_config_paths


class TestConfigSchemas:
    """Test suite for validating Hydra configuration schemas for pose estimation."""
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Clear Hydra state before and after each test."""
        GlobalHydra.instance().clear()
        yield
        GlobalHydra.instance().clear()

    def test_config_files_exist(self):
        """Test that all required config files exist."""
        config_paths = get_mock_config_paths()
        for name, path in config_paths.items():
            assert Path(path).exists(), f"Config file {name} not found at {path}"

    def test_config_files_are_valid_yaml(self):
        """Test that all config files are valid YAML."""
        config_paths = get_mock_config_paths()
        for name, path in config_paths.items():
            try:
                config = OmegaConf.load(path)
                assert isinstance(config, DictConfig)
            except Exception as e:
                pytest.fail(f"Failed to load {name} as YAML: {e}")

    def test_model_config_schema(self):
        """Test pose model config has required keys."""
        config_paths = get_mock_config_paths()
        config = OmegaConf.load(config_paths["model_config"])
        
        required_keys = {"_target_"}
        assert required_keys.issubset(config.keys()), f"Model config missing keys: {required_keys - set(config.keys())}"
        assert "yolov8" in config["_target_"].lower(), "Model should be YOLOv8-based"

    def test_infer_config_composition(self):
        """Test main inference config can be composed with Hydra."""
        if Path("/workspace").exists():
            config_dir = Path("/workspace/configs/infer/pose").resolve()
        else:
            config_dir = Path("/mnt/c/Users/kamim/code/tennis_systems/configs/infer/pose").resolve()
        
        assert config_dir.exists(), f"Config directory not found: {config_dir}"
        
        try:
            with initialize_config_dir(config_dir=str(config_dir), version_base=None):
                cfg = compose(config_name="default.yaml")
                assert cfg is not None
                assert isinstance(cfg, DictConfig)
                
                essential_sections = {"model", "pipeline"}
                assert essential_sections.issubset(cfg.keys()), f"Config missing sections: {essential_sections - set(cfg.keys())}"
        except Exception as e:
            pytest.fail(f"Failed to compose main inference config: {e}")
