import torch


def positional_encoding(tensor, num_encoding_functions=6, include_input=True, log_sampling=True) -> torch.Tensor:
    frequency_bands = 2. ** torch.linspace(0., num_encoding_functions - 1, num_encoding_functions).to(tensor)

    mul_frequencies = torch.einsum('ij,k->ijk', tensor, frequency_bands)
    sin_frequencies = torch.sin(mul_frequencies)
    cos_frequencies = torch.cos(mul_frequencies)

    full_frequencies = torch.cat([sin_frequencies, cos_frequencies], dim=-1)

    if include_input:
        broadcastable_tensor = tensor.reshape(*tensor.shape, 1)
        full_frequencies = torch.cat([broadcastable_tensor, full_frequencies], dim=-1)

    last_index_shape = 2 * num_encoding_functions * tensor.shape[-1] + include_input * tensor.shape[-1]
    full_frequencies = full_frequencies.reshape(-1, last_index_shape)
    return full_frequencies