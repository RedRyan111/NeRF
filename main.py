import torch
from tqdm import tqdm
from data_manager import DataManager
from display_helper import display_image, create_video, save_image
from models.small_NeRF_model import SmallNerfModel
from nerf_forward_pass import NeRFManager
from positional_encoding import positional_encoding
from query_points import QueryPointSamplerFromRays
from ray_bundle import RayOriginAndDirectionManager
from setup import set_random_seeds, load_training_config_yaml, get_tensor_device


set_random_seeds()
training_config = load_training_config_yaml()
device = get_tensor_device()
data_manager = DataManager('tiny_nerf_data.npz', device=device)

# training parameters
lr = training_config['training_variables']['learning_rate']
num_iters = training_config['training_variables']['num_iters']
num_encoding_functions = training_config['positional_encoding']['num_encoding_functions']

# Misc parameters
display_every = training_config['display_variables']['display_every']

# Specify encoding function.
encode = lambda x: positional_encoding(x, num_encoding_functions=num_encoding_functions)

# Initialize model and optimizer
model = SmallNerfModel(num_encoding_functions=num_encoding_functions).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Setup classes
query_sampler = QueryPointSamplerFromRays(training_config)
ray_origin_and_direction_manager = RayOriginAndDirectionManager(data_manager, device)
NeRF_manager = NeRFManager(data_manager, encode, ray_origin_and_direction_manager, query_sampler, model)

psnrs = []
test_img, test_pose = data_manager.get_random_image_and_pose_example()
for i in tqdm(range(num_iters)):

    target_img, target_tform_cam2world = data_manager.get_image_and_pose(i)

    rgb_predicted = NeRF_manager.forward(target_tform_cam2world)

    loss = torch.nn.functional.mse_loss(rgb_predicted, target_img)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if i % display_every == 0:
        psnr = -10. * torch.log10(loss)
        psnrs.append(psnr.item())

        print("Loss:", loss.item())
        display_image(i, display_every, psnrs, rgb_predicted)

    if i == num_iters-1:
        save_image(i, display_every, psnrs, rgb_predicted)
        create_video(NeRF_manager, device)

print('Done!')

