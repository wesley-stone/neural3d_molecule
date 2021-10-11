from .trainer import train_homolumo


def call(cfg):
    if cfg.task == 'train':
        train_homolumo(cfg)
    pass