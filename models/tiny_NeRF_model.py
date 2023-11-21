import torch


class TinyNerfModel(torch.nn.Module):
    def __init__(self, num_encoding_functions, filter_size=128):
        super(TinyNerfModel, self).__init__()
        # Input layer (default: 39 -> 128)
        self.layer1 = torch.nn.Linear(3 + 3 * 2 * num_encoding_functions, filter_size)
        # Layer 2 (default: 128 -> 128)
        self.layer2 = torch.nn.Linear(filter_size, filter_size)
        # Layer 3 (default: 128 -> 4)
        self.layer3 = torch.nn.Linear(filter_size, 4)
        # Short hand for torch.nn.functional.relu
        self.relu = torch.nn.functional.relu
        self.sig = torch.sigmoid
        self.rgb_layer = torch.nn.Linear(filter_size, 3)
        self.density = torch.nn.Linear(filter_size, 1)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        # x = self.layer3(x)
        rgb = self.sig(self.rgb_layer(x))
        density = self.sig(self.density(x))*10
        return rgb, torch.squeeze(density)
