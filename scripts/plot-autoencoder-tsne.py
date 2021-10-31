import argparse
import torch
import numpy as np
import pickle
import os

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

from src.models.autoencoder import WaterAutoEncoder

parser = argparse.ArgumentParser()

parser.add_argument("--model", required=True)
parser.add_argument("--dataset-folder", required=True)

args = parser.parse_args()

model = WaterAutoEncoder.load_from_checkpoint(args.model)
train_dataset = pickle.load(open(os.path.join(args.dataset_folder, "train.pkl"), "rb"))
val_dataset = pickle.load(open(os.path.join(args.dataset_folder, "val.pkl"), "rb"))
test_dataset = pickle.load(open(os.path.join(args.dataset_folder, "test.pkl"), "rb"))


features = []
features_colors = []
for X in train_dataset["X"]:
    X = torch.tensor(X).float()
    X = X.unsqueeze(0)
    X_features = model.encoder(X)[0]
    
    features.append(X_features.cpu().detach().numpy())
    features_colors.append("r")

for X in val_dataset["X"]:
    X = torch.tensor(X).float()
    X = X.unsqueeze(0)
    X_features = model.encoder(X)[0]
    
    features.append(X_features.cpu().detach().numpy())
    features_colors.append("b")

for X in test_dataset["X"]:
    X = torch.tensor(X).float()
    X = X.unsqueeze(0)
    X_features = model.encoder(X)[0]
    
    features.append(X_features.cpu().detach().numpy())
    features_colors.append("g")

features = np.stack(features)

scaler = StandardScaler()
features = scaler.fit_transform(features)
print("[!] Fitting T-SNE")
tsne = TSNE(n_components=2)
features = tsne.fit_transform(features)

plt.scatter(features[:, 0], features[:, 1], c=features_colors)

plt.legend()
plt.show()