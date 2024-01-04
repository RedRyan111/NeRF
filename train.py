import torch
from tqdm import tqdm
from data_loaders.tiny_data_loader import DataLoader
# from data_loaders.lego_data_loader import DataLoader
from display_utils.display_helper import display_image, create_video
from NeRF.models.full_model import NerfModel
from NeRF.nerf_forward_pass import EncodedModelInputs, ModelIteratorOverRayChunks
from NeRF.positional_encoding import PositionalEncoding
from NeRF.sample_points_from_rays import PointSamplerFromRays
from NeRF.rays_from_camera_builder import RaysFromCameraBuilder
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
chunksize = training_config['rendering_variables']['samples_per_model_forward_pass']

# Misc parameters
display_every = training_config['display_variables']['display_every']

# Specify encoding classes
position_encoder = PositionalEncoding(3, num_positional_encoding_functions, True)
direction_encoder = PositionalEncoding(3, num_directional_encoding_functions, True)

# Initialize model and optimizer
model = NerfModel(num_positional_encoding_functions, num_directional_encoding_functions).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Setup ray classes
point_sampler = PointSamplerFromRays(training_config)
rays_from_camera_builder = RaysFromCameraBuilder(data_manager, device)

encoded_model_inputs = EncodedModelInputs(position_encoder,
                                          direction_encoder,
                                          rays_from_camera_builder,
                                          point_sampler,
                                          depth_samples_per_ray)


psnrs = []
for i in tqdm(range(num_iters)):

    target_img, target_tform_cam2world = data_manager.get_image_and_pose(i)

    encoded_points_on_ray, encoded_ray_directions, depth_values = encoded_model_inputs.encoded_points_and_directions_from_camera(target_tform_cam2world)

    model_forward_iterator = ModelIteratorOverRayChunks(chunksize, encoded_points_on_ray, encoded_ray_directions, depth_values,
                                                        target_img, model)

    predicted_image = []
    loss_sum = 0
    for predicted_pixels, target_pixels in model_forward_iterator:
        loss = torch.nn.functional.mse_loss(predicted_pixels, target_pixels)
        loss.backward()
        loss_sum += loss.detach()

        predicted_image.append(predicted_pixels)

    optimizer.step()
    optimizer.zero_grad()
    model.zero_grad()

    predicted_image = torch.concatenate(predicted_image, dim=0).reshape(target_img.shape[0], target_img.shape[1], 3)

    if i % display_every == 0:
        psnr = -10. * torch.log10(loss_sum)
        psnrs.append(psnr.item())

        print("Loss:", loss_sum)
        display_image(i, display_every, psnrs, predicted_image, target_img)

    if i == num_iters - 1:
        #save_image(display_every, psnrs, predicted_image)
        create_video()

print('Done!')
