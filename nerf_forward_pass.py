import torch
from torch import nn
from tqdm import tqdm

from render_volume_density import render_volume_density


class NeRFManager(nn.Module):
    def __init__(self, pos_encoding_function, dir_encoding_function, rays_from_camera_builder, query_sampler, depth_samples_per_ray):
        super().__init__()
        self.pos_encoding_function = pos_encoding_function
        self.dir_encoding_function = dir_encoding_function
        self.rays_from_camera_builder = rays_from_camera_builder
        self.query_sampler = query_sampler
        self.depth_samples_per_ray = depth_samples_per_ray
        #self.input_dim = 3 + 3 * 2 * num_encoding_functions

    def forward(self, model, tform_cam2world, target_img, optimizer):

        ray_origins, ray_directions = self.rays_from_camera_builder.ray_origins_and_directions_from_pose(tform_cam2world)

        query_points, depth_values = self.query_sampler.compute_query_points_from_rays(ray_origins, ray_directions)

        depth_values = depth_values.reshape(-1, self.query_sampler.depth_samples_per_ray)

        ray_dir_new_shape = (target_img.shape[0], target_img.shape[1], 1, 3)
        ray_directions = ray_directions.reshape(ray_dir_new_shape).expand(query_points.shape)

        encoded_query_points = self.pos_encoding_function(query_points)
        encoded_ray_directions = self.dir_encoding_function(ray_directions)

        encoded_points_example = encoded_query_points.reshape(target_img.shape[0] * target_img.shape[1], self.query_sampler.depth_samples_per_ray, -1)
        encoded_ray_directions = encoded_ray_directions.reshape(target_img.shape[0] * target_img.shape[1], self.query_sampler.depth_samples_per_ray, -1)

        #make another data loader?
        rgb_predicted = []
        loss_sum = 0
        chunksize = 9000
        num_of_chunks = encoded_points_example.shape[0] // chunksize
        for j in range(0, encoded_points_example.shape[0], chunksize): #dont think this gets the final chunks...

            encoded_points = encoded_points_example[j: j+chunksize]
            encoded_ray_origins = encoded_ray_directions[j: j+chunksize]

            rgb, density = model(encoded_points, encoded_ray_origins)

            cur_depth_values = depth_values[j: j+chunksize]

            cur_rgb_predicted = render_volume_density(rgb, density, cur_depth_values)

            loss = torch.nn.functional.mse_loss(cur_rgb_predicted, target_img.reshape(-1, 3)[j: j+chunksize]) / num_of_chunks
            loss_sum += loss.detach()
            loss.backward()

            rgb_predicted.append(cur_rgb_predicted)

        optimizer.step()
        optimizer.zero_grad()
        model.zero_grad()

        rgb_predicted = torch.concatenate(rgb_predicted, dim=0).reshape(target_img.shape[0], target_img.shape[1], 3)
        return rgb_predicted, loss_sum
