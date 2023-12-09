import torch
from torch import nn
from tqdm import tqdm

from render_volume_density import render_volume_density


class NeRFManager(nn.Module):
    def __init__(self, pos_encoding_function, dir_encoding_function, rays_from_camera_builder, query_sampler,
                 depth_samples_per_ray):
        super().__init__()
        self.pos_encoding_function = pos_encoding_function
        self.dir_encoding_function = dir_encoding_function
        self.rays_from_camera_builder = rays_from_camera_builder
        self.query_sampler = query_sampler
        self.depth_samples_per_ray = depth_samples_per_ray

    def forward(self, model, tform_cam2world, target_img, optimizer):
        image_height = target_img.shape[0]  # TODO: Move these
        image_width = target_img.shape[1]

        ray_origins, ray_directions = self.rays_from_camera_builder.ray_origins_and_directions_from_pose(
            tform_cam2world)

        query_points, depth_values = self.query_sampler.compute_query_points_from_rays(ray_origins, ray_directions)

        print(f'query points: {query_points.shape} depth values: {depth_values.shape} ray_directions: {ray_directions.shape}')

        depth_values = depth_values.reshape(-1, self.query_sampler.depth_samples_per_ray)

        ray_dir_new_shape = (image_height, image_width, 1, 3)
        ray_directions = ray_directions.reshape(ray_dir_new_shape).expand(query_points.shape)  # turn into function

        #lots of reshapes that could possibly be condensed

        encoded_query_points = self.pos_encoding_function.forward(query_points)
        encoded_ray_directions = self.dir_encoding_function.forward(ray_directions)

        print(f'encoded query points: {encoded_query_points.shape} encoded ray directions: {encoded_ray_directions.shape}')

        #break everything above into its own clasS?

        rgb_predicted = []
        loss_sum = 0
        chunksize = 9000
        model_forward_iterator = ModelForward(chunksize, encoded_query_points, encoded_ray_directions, depth_values,
                                              target_img, model)
        num_of_chunks = image_height * image_width * self.query_sampler.depth_samples_per_ray / 9000 #maybe not neccessary?

        for predicted_rgb, target_img in model_forward_iterator:
            loss = torch.nn.functional.mse_loss(predicted_rgb, target_img) / num_of_chunks
            loss_sum += loss.detach()
            loss.backward()

            rgb_predicted.append(predicted_rgb)

        optimizer.step()
        optimizer.zero_grad()
        model.zero_grad()

        rgb_predicted = torch.concatenate(rgb_predicted, dim=0).reshape(image_height, image_width, 3)
        return rgb_predicted, loss_sum


class ModelForward(object):
    def __init__(self, chunk_size, encoded_query_points, encoded_ray_directions, depth_values, target_image, model):
        self.chunk_size = chunk_size
        self.chunk_index = -1
        self.num_of_rays = encoded_ray_directions.shape[0]

        self.encoded_query_points = torch.split(encoded_query_points, chunk_size)
        self.encoded_ray_directions = torch.split(encoded_ray_directions, chunk_size)
        self.depth_values = torch.split(depth_values, chunk_size)
        self.target_image = torch.split(target_image.reshape(-1, 3), chunk_size)

        self.model = model

    def __iter__(self):
        return self

    def is_out_of_bounds(self):
        return (self.chunk_index + 1) * self.chunk_size > self.num_of_rays

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
