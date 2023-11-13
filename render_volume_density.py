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
        ray_origins: torch.Tensor,
        depth_values: torch.Tensor
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    r"""Differentiably renders a radiance field, given the origin of each ray in the
    "bundle", and the sampled depth values along them.

    Args:
      radiance_field (torch.Tensor): A "field" where, at each query location (X, Y, Z),
        we have an emitted (RGB) color and a volume volume_density (denoted :math:`\sigma` in
        the paper) (shape: :math:`(width, height, num_samples, 4)`).
      ray_origins (torch.Tensor): Origin of each ray in the "bundle" as returned by the
        `get_ray_bundle()` method (shape: :math:`(width, height, 3)`).
      depth_values (torch.Tensor): Sampled depth values along each ray
        (shape: :math:`(num_samples)`).

    Returns:
      rgb_map (torch.Tensor): Rendered RGB image (shape: :math:`(width, height, 3)`).
      depth_map (torch.Tensor): Rendered depth image (shape: :math:`(width, height)`).
      acc_map (torch.Tensor): # TODO: Double-check (I think this is the accumulated
        transmittance map).
    """
    print(f'depth values: {depth_values.shape}')
    print(f'')
    # TESTED
    volume_density = torch.nn.functional.relu(radiance_field[..., 3])
    rgb = torch.sigmoid(radiance_field[..., :3])

    dists = get_differences_in_depth_values(depth_values)

    print(f'volume_density: {volume_density.shape} dist: {dists.shape}')
    alpha = 1. - torch.exp(-volume_density * dists)
    weights = alpha * cumprod_exclusive(1. - alpha + 1e-10)
    print(f'alpha: {alpha.shape} cumprod: {cumprod_exclusive(1. - alpha + 1e-10).shape}')

    # sum (volume_density * rgb * (integral of previous volume_density))

    rgb_map = (weights[..., None] * rgb).sum(dim=-2)
    depth_map = (weights * depth_values).sum(dim=-1)
    acc_map = weights.sum(-1)
    print('-----------------------------------------------')
    return rgb_map, depth_map, acc_map


def get_differences_in_depth_values(depth_values):
    final_depth_value = torch.tensor([1e10]).to(depth_values).expand(depth_values[..., :1].shape)
    depth_values = torch.cat((depth_values, final_depth_value), dim=-1)
    dists = depth_values[..., 1:] - depth_values[..., :-1]
    return dists


def composited_color_weights(volume_density, dists):
    alpha = 1. - torch.exp(-volume_density * dists)
    weights = alpha * cumprod_exclusive(1. - alpha + 1e-10)
    return weights

