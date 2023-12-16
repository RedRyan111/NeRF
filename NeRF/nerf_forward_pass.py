import torch

from NeRF.render_volume_density import render_volume_density


class EncodedModelInputs:
    def __init__(self, position_encoder, direction_encoder, rays_from_camera_builder, point_sampler,
                 depth_samples_per_ray):
        super().__init__()
        self.pos_encoding_function = position_encoder
        self.dir_encoding_function = direction_encoder
        self.rays_from_camera_builder = rays_from_camera_builder
        self.point_sampler = point_sampler
        self.depth_samples_per_ray = depth_samples_per_ray

    def encoded_points_and_directions_from_camera(self, tform_cam2world):
        ray_origins, ray_directions = self.rays_from_camera_builder.ray_origins_and_directions_from_pose(
            tform_cam2world)

        query_points, depth_values = self.point_sampler.query_points_on_rays(ray_origins, ray_directions)

        ray_directions = expand_ray_directions_to_fit_ray_query_points(ray_directions, query_points)

        encoded_query_points = self.pos_encoding_function.forward(query_points)
        encoded_ray_directions = self.dir_encoding_function.forward(ray_directions)

        return encoded_query_points, encoded_ray_directions, depth_values


class ModelIteratorOverRayChunks(object):
    def __init__(self, chunk_size, encoded_query_points, encoded_ray_directions, depth_values, target_image, model):
        self.chunk_size = chunk_size
        self.chunk_index = -1
        self.num_of_rays = encoded_ray_directions.shape[0]

        self.encoded_query_points = torch.split(encoded_query_points, chunk_size)
        self.encoded_ray_directions = torch.split(encoded_ray_directions, chunk_size)

        self.depth_values = torch.split(depth_values.reshape(self.num_of_rays, -1), chunk_size)
        self.target_image = torch.split(target_image.reshape(-1, 3), chunk_size)

        self.model = model

    def __iter__(self):
        return self

    def is_out_of_bounds(self):
        return (self.chunk_index + 1) * self.chunk_size >= self.num_of_rays

    def __next__(self):
        if self.is_out_of_bounds():
            raise StopIteration

        self.chunk_index += 1
        encoded_points = self.encoded_query_points[self.chunk_index]
        encoded_ray_origins = self.encoded_ray_directions[self.chunk_index]

        depth_values = self.depth_values[self.chunk_index]

        rgb, density = self.model(encoded_points, encoded_ray_origins)

        rgb_predicted = render_volume_density(rgb, density, depth_values)

        return rgb_predicted, self.target_image[self.chunk_index]


def expand_ray_directions_to_fit_ray_query_points(ray_directions, query_points):
    ray_dir_new_shape = (query_points.shape[0], query_points.shape[1], 1, 3)
    ray_directions = ray_directions.reshape(ray_dir_new_shape).expand(query_points.shape)

    return ray_directions
