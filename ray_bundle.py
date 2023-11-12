import torch


class CameraToWorldSpatialTransformationManager:
    def __init__(self, spatial_matrix):
        self.spatial_matrix = spatial_matrix
        self.orientation = spatial_matrix[:3, :3]
        self.translation = spatial_matrix[:3, -1]

    def transform_ray_bundle(self, ray_bundle):
        return torch.matmul(ray_bundle, self.orientation.T)

    def expand_origin_to_match_ray_bundle_shape(self, ray_bundle):
        return self.translation.expand(ray_bundle.shape)


def get_ray_bundle(height: int, width: int, focal_length: float, tform_cam2world: torch.Tensor):
    print(f'bundle input: {tform_cam2world.shape}')
    cam2world = CameraToWorldSpatialTransformationManager(tform_cam2world)

    row_meshgrid, col_meshgrid = torch.meshgrid(
        unit_torch_arange(height, focal_length, tform_cam2world),
        unit_torch_arange(width, focal_length, tform_cam2world)
    )

    directions = get_ray_directions_from_meshgrid(row_meshgrid, col_meshgrid)

    ray_directions = cam2world.transform_ray_bundle(directions)
    ray_origins = cam2world.expand_origin_to_match_ray_bundle_shape(ray_directions)

    return ray_origins, ray_directions


def get_ray_directions_from_meshgrid(row_meshgrid, col_meshgrid):
    directions = torch.stack([
        col_meshgrid,
        -1 * row_meshgrid,
        -torch.ones_like(row_meshgrid)
    ], dim=-1)

    return directions


def unit_torch_arange(full_range, focal_length, device):
    bound = .5 * full_range / focal_length
    return torch.arange(-1 * bound, bound, 1 / focal_length).to(device)

