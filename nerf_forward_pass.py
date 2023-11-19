from torch import nn
from render_volume_density import render_volume_density


class NeRFManager(nn.Module):
    def __init__(self, encoding_function, rays_from_camera_builder, query_sampler, model):
        super().__init__()
        self.encoding_function = encoding_function
        self.rays_from_camera_builder = rays_from_camera_builder
        self.query_sampler = query_sampler
        self.model = model

    def forward(self, tform_cam2world):

        ray_origins, ray_directions = self.rays_from_camera_builder.ray_origins_and_directions_from_pose(tform_cam2world)

        query_points, depth_values = self.query_sampler.compute_query_points_from_rays(ray_origins, ray_directions)

        encoded_points = self.encoding_function(query_points)

        rgb, density = self.model(encoded_points)

        rgb_predicted = render_volume_density(rgb, density, depth_values)

        return rgb_predicted
