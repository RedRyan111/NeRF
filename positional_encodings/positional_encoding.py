import torch


def positional_encoding(tensor, num_encoding_functions=6, include_input=True, log_sampling=True) -> torch.Tensor:
    frequency_bands = 2. ** torch.linspace(0., num_encoding_functions - 1, num_encoding_functions).to(tensor)

    mul_frequencies = torch.einsum('ijkp,l->ijkpl', tensor, frequency_bands)
    sin_frequencies = torch.sin(mul_frequencies)
    cos_frequencies = torch.cos(mul_frequencies)

    full_frequencies = torch.cat([sin_frequencies, cos_frequencies], dim=-1)

    if include_input:
        broadcastable_tensor = tensor.reshape(*tensor.shape, 1)
        full_frequencies = torch.cat([broadcastable_tensor, full_frequencies], dim=-1)

    full_frequencies = full_frequencies.reshape(tensor.shape[0], tensor.shape[1], tensor.shape[2], -1)
    return full_frequencies


class PositionalEncoding:
    def __init__(self, num_dim, num_encoding_functions, include_input=True):
        self.num_encoding_functions = num_encoding_functions
        self.include_input = include_input
        self.encoding_output_dim = 2 * num_encoding_functions * num_dim + (include_input * num_dim)

    def forward(self, tensor):
        frequency_bands = 2. ** torch.linspace(0., self.num_encoding_functions - 1,
                                               self.num_encoding_functions).to(tensor)

        mul_frequencies = torch.einsum('ijkp,l->ijkpl', tensor, frequency_bands)
        sin_frequencies = torch.sin(mul_frequencies)
        cos_frequencies = torch.cos(mul_frequencies)

        full_frequencies = torch.cat([sin_frequencies, cos_frequencies], dim=-1)

        if self.include_input:
            broadcastable_tensor = tensor.reshape(*tensor.shape, 1)
            full_frequencies = torch.cat([broadcastable_tensor, full_frequencies], dim=-1)

        full_frequencies = full_frequencies.reshape(tensor.shape[0] * tensor.shape[1], tensor.shape[2], self.encoding_output_dim)
        return full_frequencies
