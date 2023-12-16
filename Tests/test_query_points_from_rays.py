import numpy as np
import torch

from NeRF.sample_points_from_rays import compute_query_points_from_rays, org_compute_query_points_from_rays
from NeRF.rays_from_camera_builder import get_ray_origins_and_directions_from_pose

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load input images, poses, and intrinsics
data = np.load("../data/tiny_nerf_data/tiny_nerf_data.npz")

# Images
images = data["images"]
# Camera extrinsics (poses)
tform_cam2world = data["poses"]
tform_cam2world = torch.from_numpy(tform_cam2world).to(device)
# Focal length (intrinsics)
focal_length = data["focal"]
focal_length = torch.from_numpy(focal_length).to(device)

# Height and width of each image
height, width = images.shape[1:3]

# Near and far clipping thresholds for depth values.
near_thresh = 2.
far_thresh = 6.

# Map images to device
images = torch.from_numpy(images[:100, ..., :3]).to(device)

print(f'cam2world: {tform_cam2world.shape}')

# Get the "bundle" of rays through all image pixels.
ray_origins, ray_directions = get_ray_origins_and_directions_from_pose(height, width, focal_length, tform_cam2world)

# Sample query points along each ray
query_points, depth_values = org_compute_query_points_from_rays(ray_origins, ray_directions, near_thresh, far_thresh, depth_samples_per_ray)
my_query_points, my_depth_values = compute_query_points_from_rays(ray_origins, ray_directions, near_thresh, far_thresh, depth_samples_per_ray)

print(f'original: ')
print(query_points)
print(f'mine: ')
print(query_points)

print(torch.prod(torch.eq(query_points, my_query_points)))
print(torch.prod(torch.eq(depth_values, my_depth_values)))
