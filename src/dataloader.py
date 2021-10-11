import os
import pandas as pd
import numpy as np
from copy import deepcopy
import torch
from typing import Counter, Iterable, Optional
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
from torch.utils.data.dataset import random_split
from pymatgen.core.periodic_table import Element
import pickle

class PickleDataset(Dataset):
    def __init__(self, data_path, min_atm, max_atm, target):
        super().__init__()
        with open(data_path, 'rb') as fp:
            data = pickle.load(fp)
            self.z = data['z']
            self.pos = data['pos']
            self.labels = data[target]
            self.idxs = [i for i in range(len(data['z'])) if len(data['z'][i]) < max_atm and len(data['z'][i]) > min_atm]
    
    def __getitem__(self, idx):
        inner_idx = self.idxs[idx]
        return Data(z=self.z[inner_idx], pos=self.pos[inner_idx], y=self.labels[inner_idx])

    def __len__(self):
        return len(self.idxs)


class HomoLumoDataset(Dataset):
    def __init__(self, xyz_path, labels, f2index:np.array) -> None:
        super().__init__()
        self.xyz_root = xyz_path
        # (data_cnt,), store HOMO and LUMO
        self.labels = torch.from_numpy(labels).type(torch.FloatTensor)
        self.f2index = deepcopy(f2index)
        
    def __getitem__(self, index):
        with open(os.path.join(self.xyz_root, f'{self.f2index[index]}.xyz'), 'r') as fp:
            txt = fp.readlines()
            z = torch.LongTensor(len(txt[2:]))
            atm_xyz = torch.zeros((len(txt[2:]), 3), dtype=torch.float32)
            for i, l in enumerate(txt[2:]):
                valid_str = [s for s in l.split(' ') if s != '']
                assert(len(valid_str) == 4)
                z[i] = Element(valid_str[0]).number
                for j in range(3):
                    # axis x, y, z
                    atm_xyz[i, j] = float(valid_str[j+1])

        # return z, atm_xyz, self.labels[index]
        return Data(z=z, pos=atm_xyz, y=self.labels[index])

    def __len__(self):
        return self.labels.shape[0]

class HomoLumoDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.label_path = cfg.homolumo.label_path
        self.xyz_path = cfg.homolumo.xyz_path
        self.batch_size = cfg.homolumo.batch_size
        self.train_val_ratio = cfg.homolumo.train_val_ratio
        self.label_xls = pd.ExcelFile(self.label_path)
        self.target = cfg.homolumo.target

    def _transform_df(self, df:pd.DataFrame):
        d = df.to_numpy()
        f2index = np.array(d[:, 0])
        labels = d[:, 1:].astype(np.float)
        # 0: HOMO
        # 1: LUMO
        if self.target == 'homo':
            labels = labels[:, 0]
        else:
            labels = labels[:, 1]
        return labels, f2index

    def prepare_data(self) -> None:
        # train&validation
        df = pd.read_excel(self.label_xls, 'Dataset')
        labels, f2index = self._transform_df(df)
        dataset = HomoLumoDataset(self.xyz_path, labels, f2index)
        train_cnt = int(len(dataset) * self.train_val_ratio)
        self.train_dataset, self.val_dataset = random_split(dataset, [train_cnt, len(dataset)-train_cnt])

        # test
        df = pd.read_excel(self.label_xls, 'TestFinal')
        labels, f2index = self._transform_df(df)
        self.test_dataset = HomoLumoDataset(self.xyz_path, labels, f2index)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=48)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=48)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=48)

        
class PickleDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.data_path = cfg.homolumo.pickle_path
        self.min_atm = cfg.homolumo.min_atm
        self.max_atm = cfg.homolumo.max_atm
        self.train_val_ratio = cfg.homolumo.train_val_ratio
        self.batch_size = cfg.homolumo.batch_size
    
    def prepare_data(self) -> None:
        train_dataset = PickleDataset(data_path=os.path.join(self.data_path, 'train.pkl'), min_atm=self.min_atm, 
                                     max_atm=self.max_atm)
        train_cnt = int(len(self.dataset) * self.train_val_ratio)
        self.train_dataset, self.val_dataset = random_split(train_dataset, [train_cnt, len(self.dataset)-train_cnt])

        self.test_dataset = PickleDataset(data_path=os.path.join(self.data_path, 'test.pkl'), min_atm=self.min_atm, 
                                     max_atm=self.max_atm)
        
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=24)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=24)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=24)
