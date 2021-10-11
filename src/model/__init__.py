from src.model.dimenet import build_dimenet
import torch

def build_model(cfg):
    if cfg.homolumo.model == 'transformer':
        
        pass
    elif cfg.homolumo.model == 'dimenet':
        build_dimenet(cfg.homolumo)