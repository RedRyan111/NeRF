import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


def display_image(iteration, display_every, psnrs, rgb_predicted, target_img):
    iternums = [i * display_every for i in range(len(psnrs))]
    plt.figure(figsize=(15, 4))
    plt.subplot(131)
    plt.imshow(target_img.detach().cpu().numpy())
    plt.title(f"Target Image")
    plt.subplot(132)
    plt.imshow(rgb_predicted.detach().cpu().numpy())
    plt.title(f"Iteration {iteration}")
    plt.subplot(133)
    plt.plot(iternums, psnrs)
    plt.title("PSNR")
    plt.show()


def save_image(display_every, psnrs, rgb_predicted, target_img):
    image_data = rgb_predicted.detach().cpu().numpy()  # Replace with your image data
    x_values = [i * display_every for i in range(len(psnrs))]  # Replace with your x values
    y_values = psnrs  # Replace with your y values

    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.imshow(target_img.detach().cpu().numpy())
    ax1.title(f"Target Image")

    # Plot the image in the first subplot
    ax2.imshow(image_data, cmap='gray')  # Use cmap='gray' if the image is grayscale
    ax2.set_title('Image')

    # Plot the scatter plot in the second subplot
    ax3.plot(x_values, y_values)
    ax3.set_title(f"Peak Signal To Noise Ratio vs Training Iteration")

    # Adjust layout to prevent clipping of titles
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig('results/Training_Result.png')


trans_t = lambda t: torch.tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1],
], dtype=torch.float32)

rot_phi = lambda phi: torch.tensor([
    [1, 0, 0, 0],
    [0, torch.cos(phi), -torch.sin(phi), 0],
    [0, torch.sin(phi), torch.cos(phi), 0],
    [0, 0, 0, 1],
], dtype=torch.float32)

rot_theta = lambda th: torch.tensor([
    [torch.cos(th), 0, -torch.sin(th), 0],
    [0, 1, 0, 0],
    [torch.sin(th), 0, torch.cos(th), 0],
    [0, 0, 0, 1],
], dtype=torch.float32)


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = torch.tensor([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=torch.float32) @ c2w
    return c2w


def get_next_frame(NeRF_manager, cam2world, angle):
    angle = torch.tensor(angle * torch.pi / 180)
    rot_matrix = torch.tensor([
        [torch.cos(angle), -torch.sin(angle), 0],
        [torch.sin(angle), torch.cos(angle), 0],
        [0, 0, 1]
    ]).to(cam2world)

    cam2world[:3, :] = torch.matmul(cam2world[:3, :].T, rot_matrix).T

    rgb_predicted = NeRF_manager.forward(cam2world)

    return rgb_predicted.detach().cpu().numpy()


def create_images(NeRF_manager, device):
    frames = []
    for th in tqdm(np.linspace(0., 360., 120, endpoint=False)):
        c2w = pose_spherical(torch.from_numpy(np.asarray(th)), torch.tensor([-30.]), torch.tensor([4.]))
        c2w = c2w.to(device)
        rgb = NeRF_manager.forward(c2w).detach().cpu().numpy()
        frames.append((255 * np.clip(rgb, 0, 1)).astype(np.uint8))
    return frames


def write_video(frames):
    size = 100, 100
    fps = 30
    out = cv2.VideoWriter('results/testing.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.write(frame)
    out.release()


def create_video(NeRF_manager, device):
    images = create_images(NeRF_manager, device)
    write_video(images)

    print(f'Done saving video')
