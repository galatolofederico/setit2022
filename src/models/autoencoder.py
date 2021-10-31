import torch
import pytorch_lightning
import torch.nn as nn
import os

from src.models.encoder import DisconnectedPathsCNNEncoder
from src.models.decoder import DisconnectedPathsCNNDecoder

class WaterAutoEncoder(pytorch_lightning.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        
        self.encoder = DisconnectedPathsCNNEncoder(self.hparams.input_channels)
        self.decoder = DisconnectedPathsCNNDecoder(self.hparams.input_channels)
        
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        features = self.encoder(x)
        reconstruction = self.decoder(features)
        
        return reconstruction
    
    def training_step(self, batch, batch_nb):
        X, _ = batch
        X_reconstructed = self.forward(X)
        
        loss = self.loss_fn(X, X_reconstructed)
        error = ((X_reconstructed - X).abs() / X).mean()

        self.log("train/loss", loss.item(), prog_bar=True)
        self.log("train/error", error.item(), prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_nb):
        X, _ = batch
        X_reconstructed = self.forward(X)
        
        loss = self.loss_fn(X, X_reconstructed)
        error = ((X_reconstructed - X).abs() / X).mean()

        self.log("validation/loss", loss.item(), prog_bar=True)
        self.log("validation/error", error.item(), prog_bar=True)
        
        return loss

    def on_test_start(self):
        self.test_errors = []

    def test_step(self, batch, batch_nb):
        X, _ = batch
        X_reconstructed = self.forward(X)
        
        error = ((X_reconstructed - X).abs() / X).mean()

        self.test_errors.append(error.item())

    def on_test_end(self):
        print("=== TEST ===")
        self.test_error = sum(self.test_errors)/len(self.test_errors)
        print(f"errors: {self.test_errors}")
        print(f"error: {self.test_error}")
        print("============")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
