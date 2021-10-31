import os
import torch
import pytorch_lightning

from src.data.dataset import WaterDataset

def train_dataloader(args, normalize):
    return torch.utils.data.DataLoader(
        WaterDataset(os.path.join(args.dataset, "train.pkl"), normalize),
        batch_size=args.batch_size,
        num_workers=os.cpu_count(),
        shuffle=True
    )

def val_dataloader(args, normalize):
    return torch.utils.data.DataLoader(
        WaterDataset(os.path.join(args.dataset, "val.pkl"), normalize),
        batch_size=args.batch_size,
        num_workers=os.cpu_count(),
        shuffle=False
    )


def test_dataloader(args, normalize):
    return torch.utils.data.DataLoader(
        WaterDataset(os.path.join(args.dataset, "test.pkl"), normalize),
        batch_size=args.batch_size,
        num_workers=os.cpu_count(),
        shuffle=False
    )
