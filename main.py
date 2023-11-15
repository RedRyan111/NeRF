import numpy as np
import torch
import matplotlib.pyplot as plt

from run_NeRF import run_one_iter_of_tinynerf
from models.very_tiny_NeRF_model import VeryTinyNerfModel
from positional_encoding import positional_encoding
# from ray_bundle import org_get_ray_bundle as get_ray_bundle
import yaml

# from query_points import org_compute_query_points_from_rays as compute_query_points_from_rays

def set_seed(seed=9458):
    torch.manual_seed(seed)
    np.random.seed(seed)

def meshgrid_xy(tensor1: torch.Tensor, tensor2: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    return torch.meshgrid(tensor1, tensor2, indexing='xy')


with open('configs/training_config.yml', 'r') as file:
    training_config = yaml.safe_load(file)

with open('configs/dataset_config.yml', 'r') as file:
    dataset_config = yaml.safe_load(file)

near_thresh = training_config['rendering_variables']['near_threshold']
far_thresh = training_config['rendering_variables']['far_threshold']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load input images, poses, and intrinsics

# Height and width of each image
height = dataset_config['image_height']
width = dataset_config['image_width']
data_file_name = dataset_config['filename']
data = np.load(data_file_name)

# Images
images = data["images"]
tform_cam2world = data["poses"]
tform_cam2world = torch.from_numpy(tform_cam2world).to(device)
focal_length = data["focal"]

print(f'focal: {focal_length}')
focal_length = torch.from_numpy(focal_length).to(device)

# Hold one image out (for test).
testimg, testpose = images[101], tform_cam2world[101]
testimg = torch.from_numpy(testimg).to(device)

# Map images to device
images = torch.from_numpy(images[:100, ..., :3]).to(device)

plt.imshow(testimg.detach().cpu().numpy())
plt.show()

"""
Parameters for TinyNeRF training
"""

num_encoding_functions = training_config['positional_encoding']['num_encoding_functions']

# Specify encoding function.
encode = lambda x: positional_encoding(x, num_encoding_functions=num_encoding_functions)

depth_samples_per_ray = training_config['rendering_variables']['depth_samples_per_ray']
chunksize = training_config['training_variables']['chunksize']

lr = training_config['training_variables']['learning_rate']
num_iters = training_config['training_variables']['num_iters']

# Misc parameters
display_every = 500  # Number of iters after which stats are displayed

model = VeryTinyNerfModel(num_encoding_functions=num_encoding_functions).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

"""
Train-Eval-Repeat!
"""

set_seed()

# Lists to log metrics etc.
psnrs = []
iternums = []

for i in range(num_iters):

    # Randomly pick an image as the target.
    target_img_idx = np.random.randint(images.shape[0])
    target_img = images[target_img_idx].to(device)
    target_tform_cam2world = tform_cam2world[target_img_idx].to(device)

    # Run one iteration of TinyNeRF and get the rendered RGB image.
    rgb_predicted = run_one_iter_of_tinynerf(model, chunksize, height, width, focal_length,
                                             target_tform_cam2world, near_thresh,
                                             far_thresh, depth_samples_per_ray,
                                             encode)

    # Compute mean-squared error between the predicted and target images. Backprop!
    loss = torch.nn.functional.mse_loss(rgb_predicted, target_img)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Display images/plots/stats
    if i % display_every == 0:
        # Render the held-out view
        rgb_predicted = run_one_iter_of_tinynerf(model, chunksize, height, width, focal_length,
                                                 testpose, near_thresh,
                                                 far_thresh, depth_samples_per_ray,
                                                 encode)
        loss = torch.nn.functional.mse_loss(rgb_predicted, target_img)
        print("Loss:", loss.item())
        psnr = -10. * torch.log10(loss)

        psnrs.append(psnr.item())
        iternums.append(i)

        plt.figure(figsize=(10, 4))
        plt.subplot(121)
        plt.imshow(rgb_predicted.detach().cpu().numpy())
        plt.title(f"Iteration {i}")
        plt.subplot(122)
        plt.plot(iternums, psnrs)
        plt.title("PSNR")
        plt.show()

print('Done!')
