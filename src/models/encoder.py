import torch
import torch.nn as nn

class DisconnectedPathsCNNEncoder(nn.Module):
    def __init__(self, paths, input_shape=(50, 41), features_size=16, l1_channels=5, l2_channels=10):
        super(DisconnectedPathsCNNEncoder, self).__init__()

        self.paths = paths
        self.input_shape = input_shape
        self.features_size = features_size
        self.l1_channels = l1_channels
        self.l2_channels = l2_channels

        self.models = nn.ModuleList()
        for i in range(0, self.paths):
            self.models.append(nn.Sequential(
                nn.BatchNorm2d(1),
                nn.Conv2d(1, self.l1_channels, 5),
                nn.MaxPool2d(3),
                nn.LeakyReLU(),
                nn.Conv2d(self.l1_channels, self.l2_channels, 5),
                nn.MaxPool2d(3),
                nn.LeakyReLU(),
            ))
        
        sentinel = torch.randn(1, 1, *self.input_shape)
        out_cnn = self.models[0](sentinel).view(-1)
        cnn_features_size = out_cnn.shape[0] * len(self.models)

        self.ff = nn.Sequential(
            nn.Linear(cnn_features_size, cnn_features_size),
            nn.LeakyReLU(),
            nn.Linear(cnn_features_size, self.features_size),
            nn.LeakyReLU(),
        )
    
    def forward(self, x):
        slices = []
        for i in range(0, self.paths):
            x_slice = x[:, i, :, :].unsqueeze(1)
            y_slice = self.models[i](x_slice)
            y_slice_flat = y_slice.view(x.shape[0], -1)

            slices.append(y_slice_flat)
        
        cnn_features = torch.stack(slices).view(x.shape[0], -1)
        features = self.ff(cnn_features)

        return features

if __name__ == "__main__":
    encoder = DisconnectedPathsCNNEncoder(3)
    x = torch.randn(5, 3, 50, 41)
    y = encoder(x)

    print(x.shape)
    print(y.shape)
    
