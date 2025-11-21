"""
Configuration loader module for the POC Early Warning System.
Provides centralized configuration management from YAML files.
"""
import os
import sys
import yaml
from src.exception import CustomException
from src.logger import logging


def read_params(config_path: str) -> dict:
    """
    Read parameters from YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        dict: Configuration dictionary

    Raises:
        CustomException: If file cannot be read or parsed
    """
    try:
        with open(config_path, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)
        return config
    except FileNotFoundError:
        raise CustomException(f"Configuration file not found: {config_path}", sys)
    except yaml.YAMLError as e:
        raise CustomException(f"Error parsing YAML file: {str(e)}", sys)
    except Exception as e:
        raise CustomException(f"Error reading configuration: {str(e)}", sys)


def get_config(config_path: str = None) -> dict:
    """
    Get configuration with validation.

    Args:
        config_path: Path to the YAML configuration file.
                    If None, uses default 'config/params.yaml'

    Returns:
        dict: Validated configuration dictionary

    Raises:
        CustomException: If configuration is invalid
    """
    if config_path is None:
        config_path = os.path.join("config", "params.yaml")

    try:
        config = read_params(config_path)
        if config is None:
            raise ValueError("Config file is empty or invalid.")

        logging.info(f"Successfully loaded configuration from {config_path}")
        return config
    except Exception as e:
        raise CustomException(f"Error loading configuration: {str(e)}", sys)
