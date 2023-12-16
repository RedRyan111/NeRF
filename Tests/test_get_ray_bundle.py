import torch

from NeRF.rays_from_camera_builder import get_ray_origins_and_directions_from_pose


def meshgrid_xy(tensor1: torch.Tensor, tensor2: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    return torch.meshgrid(tensor1, tensor2, indexing='xy')


def original_get_ray_bundle(height: int, width: int, focal_length: float, tform_cam2world: torch.Tensor):
    r"""Compute the bundle of rays passing through all pixels of an image (one ray per pixel).

    Args:
      height (int): Height of an image (number of pixels).
      width (int): Width of an image (number of pixels).
      focal_length (float or torch.Tensor): Focal length (number of pixels, i.e., calibrated intrinsics).
      tform_cam2world (torch.Tensor): A 6-DoF rigid-body transform (shape: :math:`(4, 4)`) that
        transforms a 3D point from the camera frame to the "world" frame for the current example.

    Returns:
      ray_origins (torch.Tensor): A tensor of shape :math:`(width, height, 3)` denoting the centers of
        each ray. `ray_origins[i][j]` denotes the origin of the ray passing through pixel at
        row index `j` and column index `i`.
        (TODO: double check if explanation of row and col indices convention is right).
      ray_directions (torch.Tensor): A tensor of shape :math:`(width, height, 3)` denoting the
        direction of each ray (a unit vector). `ray_directions[i][j]` denotes the direction of the ray
        passing through the pixel at row index `j` and column index `i`.
        (TODO: double check if explanation of row and col indices convention is right).
    """
    # TESTED
    ii, jj = meshgrid_xy(
        torch.arange(width).to(tform_cam2world),
        torch.arange(height).to(tform_cam2world)
    )

    print(f'ii meshgrid: {ii.shape}')

    directions = torch.stack([(ii - width * .5) / focal_length,
                              -(jj - height * .5) / focal_length,
                              -torch.ones_like(ii)
                              ], dim=-1)

    print(f'')
    ray_directions = torch.sum(directions[..., None, :] * tform_cam2world[:3, :3], dim=-1)
    ray_origins = tform_cam2world[:3, -1].expand(ray_directions.shape)
    return ray_origins, ray_directions


#maybe rescaled meshgrid can be placed into the torch.arange for faster and cleaner meshgrid creation
#something like: length = (.5 * height / focal_length) row_meshgrid = torch.arange(length, -1*length)

height = 4
width = 5
focal_length = 128
c2w = torch.tensor([[1, 2, 3, 2], [1, 4, 2, 1], [2, 5, 3, 1], [0, 0, 0, 1]], dtype=torch.float32)

c2w1 = torch.tensor([[1, 2, 3, 2], [1, 4, 2, 1], [2, 5, 3, 1], [0, 0, 0, 1]], dtype=torch.float32)
c2w2 = torch.tensor([[1, 2, 3, 2], [1, 4, 2, 1], [2, 5, 3, 1], [0, 0, 0, 1]], dtype=torch.float32)
c2w_full = torch.stack([c2w1, c2w2], dim=0)
print(f'full: {c2w_full.shape}')

org_o, org_d = original_get_ray_bundle(height, width, focal_length, c2w_full)
my_o, my_d = get_ray_origins_and_directions_from_pose(height, width, focal_length, c2w_full)

print(f'original: ')
print(org_d)
print(f'mine: ')
print(my_d)

print(torch.prod(torch.eq(org_o, my_o)))
print(torch.prod(torch.eq(org_d, my_d)))

