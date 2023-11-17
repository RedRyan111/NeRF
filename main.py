import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from data_manager import DataManager
from nerf_forward_pass import nerf_forward_pass
from models.small_NeRF_model import SmallNerfModel
from positional_encoding import positional_encoding
import yaml
import random


def display_image(psnrs, rgb_predicted):
    iternums = [i for i in range(len(psnrs))]
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.imshow(rgb_predicted.detach().cpu().numpy())
    plt.title(f"Iteration {i}")
    plt.subplot(122)
    plt.plot(iternums, psnrs)
    plt.title("PSNR")
    plt.show()


def set_seed(seed=9458):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


set_seed()

with open('configs/training_config.yml', 'r') as file:
    training_config = yaml.safe_load(file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
loss = 0
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
