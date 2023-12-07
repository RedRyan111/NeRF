import numpy as np
import torch
import random


class DataLoader:
    def __init__(self, device):
        self.data = np.load('data/tiny_nerf_data/tiny_nerf_data.npz')
        self.images = torch.from_numpy(self.data['images']).to(device)
        self.poses = torch.from_numpy(self.data['poses']).to(device)
        self.focal = torch.from_numpy(self.data['focal']).to(device)

        self.directions = np.stack([np.sum([0, 0, -1] * pose[:3, :3], axis=-1) for pose in self.data['poses']])
        self.origins = self.data['poses'][:, :3, -1]

        self.num_of_images = self.images.shape[0]
        self.image_height = self.images.shape[1]
        self.image_width = self.images.shape[2]

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

