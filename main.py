import torch
from tqdm import tqdm
from data_loaders.tiny_data_loader import DataLoader
# from data_loaders.lego_data_loader import DataLoader
from display_helper import display_image, create_video, save_image
from models.medium_NeRF_model import MediumNerfModel
from nerf_forward_pass import PointAndDirectionSampler, ModelForward
from positional_encodings.positional_encoding import PositionalEncoding
from query_point_sampler_from_rays import PointSamplerFromRays
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

# Specify encoding classes
position_encoder = PositionalEncoding(3, num_positional_encoding_functions, True)
direction_encoder = PositionalEncoding(3, num_directional_encoding_functions, True)

# Initialize model and optimizer
model = MediumNerfModel(num_positional_encoding_functions, num_directional_encoding_functions).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Setup classes
query_sampler = PointSamplerFromRays(training_config)
rays_from_camera_builder = RaysFromCameraBuilder(data_manager, device)

point_and_direction_sampler = PointAndDirectionSampler(position_encoder,
                                                       direction_encoder,
                                                       rays_from_camera_builder,
                                                       query_sampler,
                                                       depth_samples_per_ray)


psnrs = []
test_img, test_pose = data_manager.get_random_image_and_pose_example() # turn this into an iterable?
for i in tqdm(range(num_iters)):

    target_img, target_tform_cam2world = data_manager.get_image_and_pose(i)

    encoded_points_on_ray, encoded_ray_directions, depth_values = point_and_direction_sampler.encoded_points_and_directions_from_camera(target_tform_cam2world)

    predicted_image = []
    loss_sum = 0
    chunksize = 9000
    model_forward_iterator = ModelForward(chunksize, encoded_points_on_ray, encoded_ray_directions, depth_values,
                                          target_img, model)

    for predicted_pixels, target_pixels in model_forward_iterator:
        loss = torch.nn.functional.mse_loss(predicted_pixels, target_pixels)
        loss_sum += loss.detach()
        loss.backward()

        predicted_image.append(predicted_pixels)

    optimizer.step()
    optimizer.zero_grad()
    model.zero_grad()

    predicted_image = torch.concatenate(predicted_image, dim=0).reshape(target_img.shape[0], target_img.shape[1], 3)

    if i % display_every == 0:
        psnr = -10. * torch.log10(loss)
        psnrs.append(psnr.item())

        print("Loss:", loss.item())
        display_image(i, display_every, psnrs, predicted_image, target_img)

    if i == num_iters - 1:
        save_image(display_every, psnrs, predicted_image)
        create_video(NeRF_manager, device)

print('Done!')
