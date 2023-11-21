import torch


def render_volume_density(rgb: torch.Tensor, density: torch.Tensor, depth_values: torch.Tensor) -> torch.Tensor:

    depth_differences = get_differences_in_depth_values(depth_values)

    transmittance = torch.exp(-density * depth_differences)
    accumulated_transmittance = cumprod_exclusive(transmittance)
    opacity = 1. - transmittance

    #rgb_map = torch.einsum('ij,ik,ijl->il', opacity, accumulated_transmittance, rgb)
    #print(f'opacity: {opacity.shape}, acc trans: {accumulated_transmittance.shape} rgb: {rgb.shape}')
    rgb_map = torch.einsum('ik,ik,ikl->il', opacity, accumulated_transmittance, rgb)

    return rgb_map


def get_differences_in_depth_values(depth_values):
    final_depth_value = torch.tensor([1e10]).to(depth_values).expand(depth_values[..., :1].shape)
    depth_values = torch.cat((depth_values, final_depth_value), dim=-1)
    depth_differences = depth_values[..., 1:] - depth_values[..., :-1]
    return depth_differences


def my_depth_differences(depth_values):
    depth_values[..., 1:] = depth_values[..., 1:] - depth_values[..., :-1]
    final_depth_value = torch.tensor([1e10]).to(depth_values).expand(depth_values[..., :1].shape)
    depth_values = torch.cat((depth_values, final_depth_value), dim=-1)
    depth_values[..., 0] = 10
    return depth_values

# for cumulative transmittance at i, we need to get the cumulative product of
# all the previous transmittance values BEFORE i,
# so on the first transmittance value, there is no transmittance value before it,
# so we need to multiply by 1 to prevent changing the initial transmittance, that is why exculsive=True


def cumprod_exclusive(tensor: torch.Tensor) -> torch.Tensor:
    #print(f'cum prod 1: {tensor.shape}')
    tensor = get_rid_of_last_index_of_last_dimension(tensor)
    cumprod = torch.cumprod(tensor, dim=-1)
    #print(f'cum prod 2: {cumprod.shape}')
    cumprod = add_ones_to_first_index_of_last_dimension(cumprod)
    #print(f'cum prod 3: {cumprod.shape}')
    return cumprod


def add_ones_to_first_index_of_last_dimension(tensor):
    ones = torch.ones((tensor.shape[0], 1)).to(tensor)
    tensor = torch.cat((ones, tensor), dim=-1)
    return tensor


def get_rid_of_last_index_of_last_dimension(tensor):
    return tensor[..., :-1]

