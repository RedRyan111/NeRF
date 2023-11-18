from torch import nn

from render_volume_density import render_volume_density


class NeRFManager(nn.Module):
    def __init__(self, data_manager, encoding_function, ray_origin_and_direction_manager, query_sampler, model):
        super().__init__()
        self.height = data_manager.image_height
        self.width = data_manager.image_width
        self.focal_length = data_manager.focal
        self.encoding_function = encoding_function
        self.ray_origin_and_direction_manager = ray_origin_and_direction_manager
        self.query_sampler = query_sampler
        self.model = model

    def forward(self, tform_cam2world):

        ray_origins, ray_directions = self.ray_origin_and_direction_manager.get_ray_origins_and_directions_from_pose(tform_cam2world)

        query_points, depth_values = self.query_sampler.compute_query_points_from_rays(ray_origins, ray_directions)

        encoded_points = self.encoding_function(query_points)

        rgb, density = self.model(encoded_points)

        rgb_predicted = render_volume_density(rgb, density, depth_values)

        return rgb_predicted
