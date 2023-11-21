import torch
from torch import nn
from tqdm import tqdm

from render_volume_density import render_volume_density


class NeRFManager(nn.Module):
    def __init__(self, encoding_function, rays_from_camera_builder, query_sampler, depth_samples_per_ray, num_encoding_functions):
        super().__init__()
        self.encoding_function = encoding_function
        self.rays_from_camera_builder = rays_from_camera_builder
        self.query_sampler = query_sampler
        self.depth_samples_per_ray = depth_samples_per_ray
        self.input_dim = 3 + 3 * 2 * num_encoding_functions

    def forward(self, model, tform_cam2world, target_img, optimizer):

        ray_origins, ray_directions = self.rays_from_camera_builder.ray_origins_and_directions_from_pose(tform_cam2world)

        query_points, depth_values = self.query_sampler.compute_query_points_from_rays(ray_origins, ray_directions)

        encoded_points = self.encoding_function(query_points)
        encoded_points_example = encoded_points.reshape(-1, self.query_sampler.depth_samples_per_ray, self.input_dim)

        rgb_predicted = []
        loss_sum = 0
        chunksize = 7000
        num_of_chunks = encoded_points_example.shape[0] // chunksize
        for j in range(0, encoded_points_example.shape[0], chunksize):
            cur_encoded_points = encoded_points_example[j: j+chunksize]

            rgb, density = model(cur_encoded_points)

            cur_depth_values = depth_values.reshape(-1, self.query_sampler.depth_samples_per_ray)[j: j+chunksize]

            cur_rgb_predicted = render_volume_density(rgb, density, cur_depth_values)

            loss = torch.nn.functional.mse_loss(cur_rgb_predicted, target_img[j: j+chunksize]) / num_of_chunks
            loss_sum += loss.detach()
            loss.backward()

            rgb_predicted.append(cur_rgb_predicted)

        optimizer.step()
        optimizer.zero_grad()
        model.zero_grad()

        rgb_predicted = torch.concatenate(rgb_predicted, dim=0).reshape(800, 800, 3)
        return rgb_predicted, loss_sum
