import torch


class NerfModel(torch.nn.Module):
    def __init__(self, num_pos_encoding_functions, num_dir_encoding_functions):
        super(NerfModel, self).__init__()
        filter_size = 200

        self.layer1 = torch.nn.Linear(3 + 3 * 2 * num_pos_encoding_functions, filter_size)
        self.layer2 = torch.nn.Linear(filter_size, filter_size)
        self.layer3 = torch.nn.Linear(filter_size, filter_size)
        self.layer4 = torch.nn.Linear(filter_size, filter_size)
        self.layer5 = torch.nn.Linear(filter_size, filter_size)

        self.layer6 = torch.nn.Linear(filter_size + 3 + 3 * 2 * num_pos_encoding_functions, filter_size)
        self.layer7 = torch.nn.Linear(filter_size, filter_size)
        self.layer8 = torch.nn.Linear(filter_size, filter_size)
        self.layer9 = torch.nn.Linear(filter_size, filter_size)

        self.layer10 = torch.nn.Linear(filter_size + 3 + 3 * 2 * num_dir_encoding_functions, filter_size)
        self.layer11 = torch.nn.Linear(filter_size, filter_size)

        self.rgb_layer = torch.nn.Linear(filter_size, 3)
        self.density_layer = torch.nn.Linear(filter_size, 1)

        self.relu = torch.nn.functional.relu
        self.sig = torch.nn.Sigmoid()

    def forward(self, x, d):
        y = self.relu(self.layer1(x))
        y = self.relu(self.layer2(y))
        y = self.relu(self.layer3(y))
        y = self.relu(self.layer4(y))
        y = self.relu(self.layer5(y))

        y = torch.concatenate((y, x), dim=-1)

        y = self.relu(self.layer6(y))
        y = self.relu(self.layer7(y))
        y = self.relu(self.layer8(y))
        y9 = self.relu(self.layer9(y))

        density = self.relu(self.density_layer(y9))
        density = torch.squeeze(density)

        y = torch.concatenate((y9, d), dim=-1)

        y = self.relu(self.layer10(y))
        y = self.relu(self.layer11(y))

        rgb = self.sig(self.rgb_layer(y))


        return rgb, density
