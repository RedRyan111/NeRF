import torch
from typing import Optional


#This is returning all zeros
class QueryPointSamplerFromRays:
    def __init__(self, training_config):
        self.near_thresh = training_config['rendering_variables']['near_threshold']
        self.far_thresh = training_config['rendering_variables']['far_threshold']
        self.depth_samples_per_ray = training_config['rendering_variables']['depth_samples_per_ray']

        self.noise_scalar = (self.far_thresh - self.near_thresh) / self.depth_samples_per_ray

    def compute_query_points_from_rays(self,
                                       ray_origins: torch.Tensor,
                                       ray_directions: torch.Tensor,
                                       randomize: Optional[bool] = True
                                       ) -> (torch.Tensor, torch.Tensor):

        depth_values = torch.linspace(self.near_thresh, self.far_thresh, self.depth_samples_per_ray).to(ray_origins)
        if randomize:
            # noise_shape = (width, height, num_samples)
            noise_shape = (ray_origins.shape[0], ray_origins.shape[1], self.depth_samples_per_ray)
            depth_values = depth_values + torch.rand(noise_shape).to(ray_origins) * self.noise_scalar

        scaled_rays_d = torch.einsum('ijk,ijl->ijkl', depth_values, ray_directions)
        ray_origins = ray_origins.reshape(ray_origins.shape[0], ray_origins.shape[1], 1, ray_origins.shape[2])
        query_points = ray_origins + scaled_rays_d
        return query_points, depth_values
