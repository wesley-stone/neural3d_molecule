import hydra
from omegaconf import DictConfig, OmegaConf
import os

from utils import setup_logging, logger
from src import call

def init(cfg):
    setup_logging()
    pass

def exit(cfg):
    pass

def print_log(cfg):
    print(OmegaConf.to_yaml(cfg))

@hydra.main(config_path="conf", config_name="config")
def run(cfg:DictConfig):
    print_log(cfg)
    init(cfg)
    logger.info(f"output path: {os.getcwd()}")
    call(cfg)
    exit(cfg)

if __name__ == "__main__":
    run()