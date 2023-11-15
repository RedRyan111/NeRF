# One iteration of TinyNeRF (forward pass).
import torch

from query_points import compute_query_points_from_rays
from ray_bundle import get_ray_bundle
from render_volume_density import render_volume_density


def run_one_iter_of_tinynerf(model, chunksize, height, width, focal_length, tform_cam2world,
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

    print(f'query points: {query_points.shape} depth values: {depth_values.shape}')

    # "Flatten" the query points.
    flattened_query_points = query_points.reshape((-1, 3))

    # Encode the query points (default: positional encoding).
    encoded_points = encoding_function(flattened_query_points)

    print(f'flattened query points: {flattened_query_points.shape} encoded query points: {encoded_points.shape}')

    # Split the encoded points into "chunks", run the model on all chunks, and
    # concatenate the results (to avoid out-of-memory issues).
    batches = torch.split(encoded_points, chunksize)

    predictions = []
    for batch in batches:
        predictions.append(model(batch))
    radiance_field_flattened = torch.cat(predictions, dim=0)

    print(f'radiance field cat: {radiance_field_flattened.shape}')

    # "Unflatten" to obtain the radiance field.
    unflattened_shape = list(query_points.shape[:-1]) + [4]
    print(f'unflattened: {unflattened_shape}')
    radiance_field = torch.reshape(radiance_field_flattened, unflattened_shape)
    print(f'radiance field fin: {radiance_field_flattened.shape}')
    # Perform differentiable volume rendering to re-synthesize the RGB image.
    rgb_predicted = render_volume_density(radiance_field, depth_values)

    return rgb_predicted