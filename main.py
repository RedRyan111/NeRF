import torch
from tqdm import tqdm
from data_manager import DataManager
from display_helper import display_image
from nerf_forward_pass import nerf_forward_pass
from models.small_NeRF_model import SmallNerfModel
from positional_encoding import positional_encoding

from setup import set_random_seeds, load_training_config_yaml, get_tensor_device

set_random_seeds()

training_config = load_training_config_yaml()

device = get_tensor_device()

data_manager = DataManager('tiny_nerf_data.npz', device=device)

# Image parameters
height = data_manager.image_height
width = data_manager.image_width
focal_length = data_manager.focal

# training parameters
near_thresh = training_config['rendering_variables']['near_threshold']
far_thresh = training_config['rendering_variables']['far_threshold']
depth_samples_per_ray = training_config['rendering_variables']['depth_samples_per_ray']
lr = training_config['training_variables']['learning_rate']
num_iters = training_config['training_variables']['num_iters']
num_encoding_functions = training_config['positional_encoding']['num_encoding_functions']

# Specify encoding function.
encode = lambda x: positional_encoding(x, num_encoding_functions=num_encoding_functions)

# Misc parameters
display_every = 500  # Number of iters after which stats are displayed

model = SmallNerfModel(num_encoding_functions=num_encoding_functions).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

psnrs = []
test_img, test_pose = data_manager.get_random_image_and_pose_example()
for i in tqdm(range(num_iters)):

    target_img, target_tform_cam2world = data_manager.get_image_and_pose(i)

    rgb_predicted = nerf_forward_pass(model, height, width, focal_length,
                                      target_tform_cam2world, near_thresh,
                                      far_thresh, depth_samples_per_ray,
                                      encode)

    loss = torch.nn.functional.mse_loss(rgb_predicted, target_img)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if i % display_every == 0:
        rgb_predicted = nerf_forward_pass(model, height, width, focal_length,
                                          test_pose, near_thresh,
                                          far_thresh, depth_samples_per_ray,
                                          encode)
        loss = torch.nn.functional.mse_loss(rgb_predicted, target_img)
        psnr = -10. * torch.log10(loss)
        psnrs.append(psnr.item())

        print("Loss:", loss.item())
        display_image(psnrs, rgb_predicted)

print('Done!')
