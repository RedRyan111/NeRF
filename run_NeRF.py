# One iteration of TinyNeRF (forward pass).
import torch

from query_points import compute_query_points_from_rays
from ray_bundle import get_ray_bundle
from render_volume_density import render_volume_density


def run_one_iter_of_tinynerf(model, height, width, focal_length, tform_cam2world,
                             near_thresh, far_thresh, depth_samples_per_ray,
                             encoding_function):
    # Get the "bundle" of rays through all image pixels.
    ray_origins, ray_directions = get_ray_bundle(height, width, focal_length, tform_cam2world)

    print(
        f'main ray origins: {ray_origins.shape} ray directions: {ray_directions.shape} cam2world: {tform_cam2world.shape}')

    # Sample query points along each ray
    query_points, depth_values = compute_query_points_from_rays(
        ray_origins, ray_directions, near_thresh, far_thresh, depth_samples_per_ray
    )

    encoded_points = encoding_function(query_points)

    radiance_field = model(encoded_points)

    rgb_predicted = render_volume_density(radiance_field, depth_values)

    return rgb_predicted
