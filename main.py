import torch
from tqdm import tqdm
from data_loaders.data_loader import DataLoader
#from data_loaders.lego_data_loader import DataLoader
from display_helper import display_image, create_video, save_image
from models.medium_NeRF_model import MediumNerfModel
from nerf_forward_pass import NeRFManager
from positional_encodings.positional_encoding import positional_encoding
from query_points import QueryPointSamplerFromRays
from ray_bundle import RaysFromCameraBuilder
from setup_utils import set_random_seeds, load_training_config_yaml, get_tensor_device


set_random_seeds()
training_config = load_training_config_yaml()
device = get_tensor_device()
data_manager = DataLoader(device)

# training parameters
lr = training_config['training_variables']['learning_rate']
num_iters = training_config['training_variables']['num_iters']
num_positional_encoding_functions = training_config['positional_encoding']['num_positional_encoding_functions']
num_directional_encoding_functions = training_config['positional_encoding']['num_directional_encoding_functions']
depth_samples_per_ray = training_config['rendering_variables']['depth_samples_per_ray']

# Misc parameters
display_every = training_config['display_variables']['display_every']

# Specify encoding function.
position_encode = lambda x: positional_encoding(x, num_positional_encoding_functions)
direction_encode = lambda x: positional_encoding(x, num_directional_encoding_functions)

# Initialize model and optimizer
model = MediumNerfModel(num_positional_encoding_functions, num_directional_encoding_functions).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Setup classes
query_sampler = QueryPointSamplerFromRays(training_config)
rays_from_camera_builder = RaysFromCameraBuilder(data_manager, device)
NeRF_manager = NeRFManager(position_encode, direction_encode, rays_from_camera_builder, query_sampler, depth_samples_per_ray)

psnrs = []
test_img, test_pose = data_manager.get_random_image_and_pose_example()
for i in tqdm(range(num_iters)):

    target_img, target_tform_cam2world = data_manager.get_image_and_pose(i)

    #target_img = target_img.reshape(-1, 3)

    rgb_predicted, loss = NeRF_manager.forward(model, target_tform_cam2world, target_img.reshape(-1, 3), optimizer)

    if i % display_every == 0:
        psnr = -10. * torch.log10(loss)
        psnrs.append(psnr.item())

        print("Loss:", loss.item())
        display_image(i, display_every, psnrs, rgb_predicted, target_img)

    if i == num_iters - 1:
        save_image(display_every, psnrs, rgb_predicted)
        create_video(NeRF_manager, device)

print('Done!')
