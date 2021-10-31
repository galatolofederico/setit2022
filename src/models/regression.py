import torch
import pytorch_lightning
import torch.nn as nn
import os

from src.models.encoder import DisconnectedPathsCNNEncoder
from src.models.decoder import DisconnectedPathsCNNDecoder

class WaterRegressionModel(pytorch_lightning.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        
        self.encoder = DisconnectedPathsCNNEncoder(
            self.hparams.input_channels,
            features_size=self.hparams.features_size,
            l1_channels=self.hparams.encoder_l1_channels,
            l2_channels=self.hparams.encoder_l2_channels
        )

        self.decoder = DisconnectedPathsCNNDecoder(
            self.hparams.output_channels,
            input_size=self.hparams.features_size,
            l1_channels=self.hparams.decoder_l1_channels,
            l2_channels=self.hparams.decoder_l2_channels
        )
        
        self.loss_fn = getattr(torch.nn, self.hparams.loss)()

        self.y_mean = self.hparams.y_mean.to(self.device)
        self.y_std = self.hparams.y_std.to(self.device)
        

    def forward(self, x, mask=None, do_not_denormalize=False):
        features = self.encoder(x)
        output = self.decoder(features)
        
        if self.hparams.normalize and (not do_not_denormalize and not self.training):
            output = self.denormalize(output)

        if mask is not None:
            output = output * mask
        
        return output
    
    def denormalize(self, y):
        y_mean = self.y_mean.to(y.device)
        y_std = self.y_std.to(y.device)

        return (y * y_std) + y_mean

    def training_step(self, batch, batch_nb):
        X, y, mask = batch
        y_hat = self.forward(X, mask=mask, do_not_denormalize=True)
        
        loss = self.loss_fn(y, y_hat) 

        if self.hparams.normalize:
            y = self.denormalize(y)
            y_hat = self.denormalize(y_hat)
        
        absolute_error = (y_hat - y).abs().mean()

        self.log("train/loss", loss.item(), prog_bar=True)
        self.log("train/absolute_error", absolute_error.item(), prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_nb):
        X, y, mask = batch
        y_hat = self.forward(X, mask=mask, do_not_denormalize=True)
        
        loss = self.loss_fn(y, y_hat) 

        if self.hparams.normalize:
            y = self.denormalize(y)
            y_hat = self.denormalize(y_hat)
        
        absolute_error = (y_hat - y).abs().mean()

        self.log("validation/loss", loss.item(), prog_bar=True)
        self.log("validation/absolute_error", absolute_error.item(), prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
