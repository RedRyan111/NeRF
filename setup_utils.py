import random
import numpy as np
import torch
import yaml


def set_random_seeds(seed=9458):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_training_config_yaml():
    with open('configs/training_config.yml', 'r') as file:
        training_config = yaml.safe_load(file)
        return training_config


def get_tensor_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
