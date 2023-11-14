import torch


def cumprod_exclusive(tensor: torch.Tensor) -> torch.Tensor:
    r"""Mimick functionality of tf.math.cumprod(..., exclusive=True), as it isn't available in PyTorch.

    Args:
      tensor (torch.Tensor): Tensor whose cumprod (cumulative product, see `torch.cumprod`) along dim=-1
        is to be computed.

    Returns:
      cumprod (torch.Tensor): cumprod of Tensor along dim=-1, mimiciking the functionality of
        tf.math.cumprod(..., exclusive=True) (see `tf.math.cumprod` for details).
    """
    # TESTED
    # Only works for the last dimension (dim=-1)
    dim = -1
    # Compute regular cumprod first (this is equivalent to `tf.math.cumprod(..., exclusive=False)`).
    cumprod = torch.cumprod(tensor, dim)
    # "Roll" the elements along dimension 'dim' by 1 element.
    cumprod = torch.roll(cumprod, 1, dim)
    # Replace the first element by "1" as this is what tf.cumprod(..., exclusive=True) does.
    cumprod[..., 0] = 1.

    return cumprod


def render_volume_density(
        radiance_field: torch.Tensor,
        depth_values: torch.Tensor
) -> (torch.Tensor, torch.Tensor, torch.Tensor):

    print(f'depth values: {depth_values.shape}')
    print(f'')
    # TESTED
    absorption_coeff = torch.nn.functional.relu(radiance_field[..., 3])
    rgb = torch.sigmoid(radiance_field[..., :3])

    depth_differences = get_differences_in_depth_values(depth_values)

    print(f'absorption_coeff: {absorption_coeff.shape} depth_differences: {depth_differences.shape}')
    transmittance = torch.exp(-absorption_coeff * depth_differences)
    opacity = 1. - transmittance #scattering coefficient
    weights = opacity * cumprod_exclusive(1. - opacity + 1e-10)
    print(f'opacity: {opacity.shape} cumprod: {cumprod_exclusive(1. - opacity + 1e-10).shape}')

    #transmittance * background color + (1- transmittance) * volume color
    # absorption_coeff * rgb * (sum of previous absorption_coeff * depth_differences)

    rgb_map = (weights[..., None] * rgb).sum(dim=-2)
    depth_map = (weights * depth_values).sum(dim=-1)
    acc_map = weights.sum(-1)
    print('-----------------------------------------------')
    return rgb_map, depth_map, acc_map


def get_differences_in_depth_values(depth_values):
    final_depth_value = torch.tensor([1e10]).to(depth_values).expand(depth_values[..., :1].shape)
    depth_values = torch.cat((depth_values, final_depth_value), dim=-1)
    depth_differences = depth_values[..., 1:] - depth_values[..., :-1]
    return depth_differences


def composited_color_weights(volume_density, dists):
    alpha = 1. - torch.exp(-volume_density * dists)
    weights = alpha * cumprod_exclusive(1. - alpha + 1e-10)
    return weights


def my_depth_differences(depth_values):
    depth_values[..., 1:] = depth_values[..., 1:] - depth_values[..., :-1]
    return depth_values
