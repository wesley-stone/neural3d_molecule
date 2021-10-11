from .dataloader import HomoLumoDataModule, HomoLumoDataset, PickleDataModule
import os
from torch import nn
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, early_stopping 

from torch_geometric.nn import DimeNet
from torch_geometric.data import Data, Batch, DataLoader
from torch_geometric.datasets import QM9
from torch_geometric.nn import SchNet

def model_factory(mtype):
    if mtype == 'dimenet_pretrain':
        data_cache_path = '/mnt/exp/data/'
        dataset = QM9(data_cache_path)
        model, _ = DimeNet.from_qm9_pretrained(root=data_cache_path, dataset=dataset, target=0)
        return model
    elif mtype == 'schnet':
        model = SchNet()
        return model
    elif mtype == 'pna':
        
        pass

class HomoLumoModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        # control learning rate by itself.
        self.automatic_optimization = False

        # input dim equals the maximum basis length.
        self.nn = model_factory(cfg.homolumo.model)
        self.loss = nn.MSELoss(reduction='mean')
        self.lr = cfg.homolumo.lr


    def _shared_step(self, batch:Batch):
        pred = self.nn(batch.z, batch.pos, batch.batch)
        pred = pred.squeeze(1)
        loss = self.loss(pred, batch.y)
        return loss, (pred, batch.y)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()
        opt.zero_grad()
        loss, _ = self._shared_step(batch)
        self.manual_backward(loss)
        opt.step()
        self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True)
        sch.step()
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=2000, gamma=0.5)
        return {
            'optimizer': opt,
            'lr_scheduler': sch,
        }
    

    def validation_step(self, batch, batch_idx):
        loss, _ = self._shared_step(batch)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        loss, (pred, y) = self._shared_step(batch)
        self.log('test_rmse', loss**0.5)
        test_mae = (pred-y).abs().mean()
        self.log('test_mae', test_mae)

def train_homolumo(cfg):
    pl.seed_everything(cfg.homolumo.seed)
    data_module = HomoLumoDataModule(cfg)
    model = HomoLumoModule(cfg)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss')
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=100)
    callbacks = [checkpoint_callback, 
                 early_stopping_callback]
    trainer = pl.Trainer(gpus=[1], callbacks=callbacks, max_epochs=1)
    trainer.fit(model, data_module)

    # test
    trainer.test()
    
def test_model(cfg):
    data_module = HomoLumoDataModule(cfg)
    model = model_factory(cfg.homolumo.model)
    z = torch.LongTensor([0, 1, 1])
    pos = torch.FloatTensor(
        [[0.1, 0.2, -0.5],
        [0.5, 0.2, -0.5],
        [0.1, 0.2, -0.1]]
                            )
    batch = torch.LongTensor([0, 0, 1])
    out = model(z, pos, batch)
    print(out.shape)

def test_dimenet(cfg):
    data_cache_path = '/mnt/exp/data/'
    dataset = QM9(data_cache_path)

    # DimeNet uses the atomization energy for targets U0, U, H, and G.
    idx = torch.tensor([0, 1, 2, 3, 4, 5, 6, 12, 13, 14, 15, 11])
    dataset.data.y = dataset.data.y[:, idx]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for target in [2, 3]:
        # Skip target \delta\epsilon, since it can be computed via
        # \epsilon_{LUMO} - \epsilon_{HOMO}.
        if target == 4:
            continue

        model, datasets = DimeNet.from_qm9_pretrained(data_cache_path, dataset, target)
        train_dataset, val_dataset, test_dataset = datasets

        model = model.to(device)
        loader = DataLoader(test_dataset, batch_size=64)

        maes = []
        for data in loader:
            data = data.to(device)
            with torch.no_grad():
                pred = model(data.z, data.pos, data.batch)
            mae = (pred.view(-1) - data.y[:, target]).abs()
            maes.append(mae)

        mae = torch.cat(maes, dim=0)

        # Report meV instead of eV.
        mae = 1000 * mae if target in [2, 3, 4, 6, 7, 8, 9, 10] else mae

        print(f'Target: {target:02d}, MAE: {mae.mean():.5f} ± {mae.std():.5f}')