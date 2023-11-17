import numpy as np
import torch
from matplotlib import pyplot as plt
import random


class DataManager:
    def __init__(self, filename, device):
        seed = 9458
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.data = np.load(filename)
        self.images = torch.from_numpy(self.data['images']).to(device)
        self.poses = torch.from_numpy(self.data['poses']).to(device)
        self.focal = torch.from_numpy(self.data['focal']).to(device)

        self.directions = np.stack([np.sum([0, 0, -1] * pose[:3, :3], axis=-1) for pose in self.data['poses']])
        self.origins = self.data['poses'][:, :3, -1]

        self.num_of_images = self.images.shape[0]
        self.image_height = self.images.shape[1]
        self.image_width = self.images.shape[2]

        #self.print_data_info()
        #self.print_example_image_data()
        #self.print_data_camera_poses_and_directions()

    def get_example_index(self):
        return random.randint(0, self.num_of_images-1)

    def get_image_and_pose(self, index):
        index = index % self.num_of_images
        image = self.images[index]
        pose = self.poses[index]
        return image, pose

    def get_random_image_and_pose_example(self):
        index = self.get_example_index()
        image = self.images[index]
        pose = self.poses[index]
        return image, pose

    def print_data_info(self):
        print(f'Images shape: {self.images.shape}')
        print(f'Poses shape: {self.poses.shape}')
        print(f'Focal length: {self.focal}')

    def print_example_image_data(self):
        rand_ind = random.randint(0, self.num_of_images-1)
        example_img = self.images[rand_ind]
        example_pose = self.poses[rand_ind]
        plt.imshow(example_img.cpu())
        print('Pose')
        print(example_pose)
        plt.show()
        plt.close()

    def print_data_camera_poses_and_directions(self):
        ax = plt.figure(figsize=(12, 8)).add_subplot(projection='3d')
        _ = ax.quiver(
          self.origins[..., 0].flatten(),
          self.origins[..., 1].flatten(),
          self.origins[..., 2].flatten(),
          self.directions[..., 0].flatten(),
          self.directions[..., 1].flatten(),
          self.directions[..., 2].flatten(), length=0.5, normalize=True)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('z')
        plt.show()
        plt.close()

