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


class RaysFromCameraBuilder:
    def __init__(self, data_manager, device):
        self.height = data_manager.image_height
        self.width = data_manager.image_width
        self.focal_length = data_manager.focal
        self.device = device

        self.directions = self.get_ray_directions().to(device)

    def get_ray_directions(self):
        row_meshgrid, col_meshgrid = torch.meshgrid(
            unit_length_torch_arange(self.height, self.focal_length).to(self.device),
            unit_length_torch_arange(self.width, self.focal_length).to(self.device)
        )

        return get_ray_directions_from_meshgrid(row_meshgrid, col_meshgrid)

    def ray_origins_and_directions_from_pose(self, camera_to_world_transform: torch.Tensor):
        cam2world = CameraToWorldSpatialTransformationManager(camera_to_world_transform)

        ray_directions = cam2world.transform_ray_bundle(self.directions)
        ray_origins = cam2world.expand_origin_to_match_ray_bundle_shape(ray_directions)

        return ray_origins, ray_directions


def get_ray_directions_from_meshgrid(row_meshgrid, col_meshgrid):
    directions = torch.stack([
        col_meshgrid,
        -1 * row_meshgrid,
        -torch.ones_like(row_meshgrid)
    ], dim=-1)

    return directions


def unit_length_torch_arange(full_range, resolution):
    bound = .5 * full_range / resolution
    return torch.arange(-1 * bound, bound, 1 / resolution)
