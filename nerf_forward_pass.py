from query_points import compute_query_points_from_rays
from ray_bundle import get_ray_origins_and_directions_from_pose
from render_volume_density import render_volume_density


def nerf_forward_pass(model, height, width, focal_length, tform_cam2world,
                      near_thresh, far_thresh, depth_samples_per_ray,
                      encoding_function):

    ray_origins, ray_directions = get_ray_origins_and_directions_from_pose(height, width, focal_length, tform_cam2world)

    query_points, depth_values = compute_query_points_from_rays(
        ray_origins, ray_directions, near_thresh, far_thresh, depth_samples_per_ray
    )

    encoded_points = encoding_function(query_points)

    rgb, density = model(encoded_points)

    rgb_predicted = render_volume_density(rgb, density, depth_values)

    return rgb_predicted
