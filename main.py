from typing import Optional

import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from data_loader import DataLoader
#from lego_data_loader import DataLoader
from display_helper import display_image, create_video, save_image
from models.small_NeRF_model import SmallNerfModel
from models.tiny_NeRF_model import TinyNerfModel
from nerf_forward_pass import NeRFManager
from positional_encoding import positional_encoding
from query_points import QueryPointSamplerFromRays
from ray_bundle import RaysFromCameraBuilder
from setup_utils import set_random_seeds, load_training_config_yaml, get_tensor_device


def get_minibatches(inputs: torch.Tensor, chunksize: Optional[int] = 2):
    return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]


set_random_seeds()
training_config = load_training_config_yaml()
device = get_tensor_device()
print(f'device: {device}')
data_manager = DataLoader('tiny_nerf_data.npz', device)
#data_manager = DataLoader(device)

print(f'data manager poses')
print(data_manager.poses.shape)
print(data_manager.poses[0])

# training parameters
lr = training_config['training_variables']['learning_rate']
num_iters = training_config['training_variables']['num_iters']
num_encoding_functions = training_config['positional_encoding']['num_encoding_functions']
depth_samples_per_ray = training_config['rendering_variables']['depth_samples_per_ray']

# Misc parameters
display_every = training_config['display_variables']['display_every']

# Specify encoding function.
encode = lambda x: positional_encoding(x, num_encoding_functions)

# Initialize model and optimizer
model = TinyNerfModel(num_encoding_functions).to(device)
#model = SmallNerfModel(num_encoding_functions).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Setup classes
query_sampler = QueryPointSamplerFromRays(training_config)
rays_from_camera_builder = RaysFromCameraBuilder(data_manager, device)
NeRF_manager = NeRFManager(encode, rays_from_camera_builder, query_sampler, depth_samples_per_ray, num_encoding_functions)

psnrs = []
test_img, test_pose = data_manager.get_random_image_and_pose_example()
for i in tqdm(range(num_iters)):

    target_img, target_tform_cam2world = data_manager.get_image_and_pose(i)

    target_img = target_img.reshape(-1, 3)

    rgb_predicted, loss = NeRF_manager.forward(model, target_tform_cam2world, target_img, optimizer)

    if i % display_every == 0:
        psnr = -10. * torch.log10(loss)
        psnrs.append(psnr.item())

        print("Loss:", loss.item())
        display_image(i, display_every, psnrs, rgb_predicted)#/255)

    if i == num_iters - 1:
        save_image(display_every, psnrs, rgb_predicted)
        create_video(NeRF_manager, device)

print('Done!')
