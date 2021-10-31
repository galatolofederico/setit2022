import os
import torch
import pickle
import json
from pathlib import Path

class WaterDataset(torch.utils.data.Dataset):
    def __init__(self, file, normalize):
        self.dataset = pickle.load(open(file, "rb"))
        self.normalize = normalize
        path = Path(file)
        self.path = path.parent.absolute()
        self.stats = json.load(open(os.path.join(self.path, "stats.json"), "r"))
        self.y_mean = torch.tensor(self.stats["y"]["mean"]).float().unsqueeze(1).unsqueeze(2)
        self.y_std = torch.tensor(self.stats["y"]["std"]).float().unsqueeze(1).unsqueeze(2)
        
        assert len(self.dataset["X"]) == len(self.dataset["y"])
        
        self.input_channels = self.dataset["X"][0].shape[0]
        self.output_channels = self.dataset["y"][0].shape[0]

    def __len__(self):
        return len(self.dataset["X"])

    def __getitem__(self, idx):
        mask = torch.tensor(self.dataset["mask"][idx]).float().unsqueeze(0)
        
        X = torch.tensor(self.dataset["X"][idx]).float()
        y = torch.tensor(self.dataset["y"][idx]).float()

        if self.normalize:
            y = (y - self.y_mean) / self.y_std

        X = X*mask
        y = y*mask

        return X, y, mask 
        

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--file", required=True)

    args = parser.parse_args()

    ds = WaterDataset(args.file)

    for i, o in ds:
        print(i.shape, o.shape)