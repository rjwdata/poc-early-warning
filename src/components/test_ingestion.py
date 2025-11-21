import os
import argparse
from src.logger import logging
from src.exception import CustomException
from src.config_loader import get_config
import pandas as pd

def main(config_path, datasource):
    config = get_config(config_path)
    logging.info(f"Configuration loaded: {config}")
    return config

if __name__ =="__main__":
    args = argparse.ArgumentParser()
    default_config_path = os.path.join("config", "params.yaml")
    args.add_argument("--config", default=default_config_path)
    args.add_argument("--datasource", default=None)

    parsed_args = args.parse_args()
    main(config_path=parsed_args.config, datasource=parsed_args.datasource)