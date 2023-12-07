import json
from PIL import Image
import numpy as np
import torch
from matplotlib import pyplot as plt
import random


class DataLoader:
    def __init__(self, device):
        self.image_height = 800
        self.image_width = 800

        self.file_path = 'data/nerf_synthetic/lego/'

        f = open(self.file_path + 'transforms_train.json')
        self.train_data = json.load(f)
        f.close()

        self.images = torch.from_numpy(self.get_images()).type(torch.FloatTensor).to(device)/255
        self.poses = torch.from_numpy(self.get_poses()).type(torch.FloatTensor).to(device)
        self.focal = 1200

        self.num_of_images = self.images.shape[0]

    def get_example_index(self):
        return random.randint(0, self.num_of_images - 1)

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

    def get_poses(self):
        main_data = self.train_data['frames']
        numpy_transform_matrix_list = []
        for mini_dict in main_data:
            transform_matrix = mini_dict['transform_matrix']
            transform_matrix_array = np.asarray(transform_matrix)
            numpy_transform_matrix_list.append(transform_matrix_array)

        return np.stack(numpy_transform_matrix_list)

    def get_images(self):
        main_data = self.train_data['frames']
        numpy_image_list = []
        for mini_dict in main_data:
            filename = mini_dict['file_path']
            image = Image.open(self.file_path + filename + '.png')
            image = image.convert('RGB')
            image_array = np.asarray(image)
            numpy_image_list.append(image_array)

        return np.stack(numpy_image_list)
