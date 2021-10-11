from torch_geometric.nn import DimeNet
from torch_geometric.data import Data, Batch, DataLoader
from torch_geometric.datasets import QM9
from torch_geometric.nn import SchNet

def build_dimenet(cfg):
    if cfg.pretrain:
        data_cache_path = cfg.data_cache_path
        dataset = QM9(data_cache_path)
        model, _ = DimeNet.from_qm9_pretrained(root=data_cache_path, dataset=dataset, target=0)
    else:
        model = DimeNet()
    
    return model