import torch


class SmallNerfModel(torch.nn.Module):
    def __init__(self, num_encoding_functions):
        super(SmallNerfModel, self).__init__()
        filter_size = 128

        self.layer1 = torch.nn.Linear(3 + 3 * 2 * num_encoding_functions, filter_size)
        self.layer2 = torch.nn.Linear(filter_size, filter_size)
        self.layer3 = torch.nn.Linear(filter_size, filter_size)

        self.layer4 = torch.nn.Linear(filter_size + 3 + 3 * 2 * num_encoding_functions, filter_size)
        self.layer5 = torch.nn.Linear(filter_size, filter_size)

        self.rgb_layer = torch.nn.Linear(filter_size, 3)
        self.density_layer = torch.nn.Linear(filter_size, 1)

        self.relu = torch.nn.functional.relu
        self.sig = torch.nn.Sigmoid()

    def forward(self, x):
        y = self.relu(self.layer1(x))
        y = self.relu(self.layer2(y))
        y = self.relu(self.layer3(y))

        y = torch.concatenate((y, x), dim=-1)

        y = self.relu(self.layer4(y))
        y = self.relu(self.layer5(y))

        rgb = self.sig(self.rgb_layer(y))
        density = self.relu(self.density_layer(y))
        density = torch.squeeze(density)

        return rgb, density
