import torch
import torch.nn as nn
import math

class DisconnectedPathsCNNDecoder(nn.Module):
    def __init__(self, paths, input_size=16, output_shape=(50, 41), l1_channels=5, l2_channels=10):
        super(DisconnectedPathsCNNDecoder, self).__init__()

        self.paths = paths
        self.input_size = input_size
        self.l1_channels = l1_channels
        self.l2_channels = l2_channels
        assert math.sqrt(self.input_size)**2 == self.input_size, "Input size should be a perfect root"
        self.input_shape = int(math.sqrt(self.input_size))
        self.output_shape = output_shape

        self.models = nn.ModuleList()
        for i in range(0, self.paths):
            self.models.append(nn.Sequential(
                nn.ConvTranspose2d(1, self.l1_channels, kernel_size=5, dilation=2),
                nn.ReLU(True),
                nn.ConvTranspose2d(self.l1_channels, self.l2_channels, kernel_size=7, dilation=2),
                nn.ReLU(True),
                nn.ConvTranspose2d(self.l2_channels, 1, kernel_size=10, dilation=3),
            ))
        
        sentinel = torch.randn(1, 1, self.input_shape, self.input_shape)
        out_cnn = self.models[0](sentinel)
        
        assert out_cnn.shape[2] > self.output_shape[0], "Impossible input_size for output_shape"
        assert out_cnn.shape[3] > self.output_shape[1], "Impossible input_size for output_shape"
        
    
    def forward(self, x):
        x = x.view(-1, 1, self.input_shape, self.input_shape)
        slices = []
        for i in range(0, self.paths):
            y_slice = self.models[i](x)
            y_slice_crop = y_slice[:, 0, :self.output_shape[0], :self.output_shape[1]]
            slices.append(y_slice_crop)
        out = torch.stack(slices)
        out = out.permute(1, 0, 2, 3)
        return out

if __name__ == "__main__":
    decoder = DisconnectedPathsCNNDecoder(10)
    x = torch.randn(5, 16)
    y = decoder(x)
    
    print(x.shape)
    print(y.shape)
    