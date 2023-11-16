import torch
from typing import Optional


def compute_query_points_from_rays(
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        near_thresh: float,
        far_thresh: float,
        num_samples: int,
        randomize: Optional[bool] = True
) -> (torch.Tensor, torch.Tensor):

    depth_values = torch.linspace(near_thresh, far_thresh, num_samples).to(ray_origins)
    if randomize:
        # noise_shape = (width, height, num_samples)
        noise_shape = (ray_origins.shape[0], ray_origins.shape[1], num_samples)
        noise_scalar = (far_thresh - near_thresh) / num_samples

        depth_values = depth_values + torch.rand(noise_shape).to(ray_origins) * noise_scalar

    # origins: torch.Size([100, 100, 3])
    # directions: torch.Size([100, 100, 3])
    # depth: torch.Size([100, 100, 32])
    # query points: torch.Size([100, 100, 32, 3])

    scaled_rays_d = torch.einsum('ijk,ijl->ijkl', depth_values, ray_directions)
    ray_origins = ray_origins.reshape(ray_origins.shape[0], ray_origins.shape[1], 1, ray_origins.shape[2])
    query_points = ray_origins + scaled_rays_d
    return query_points, depth_values

