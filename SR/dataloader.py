import os
import numpy as np
import cv2
import random
from pathlib import Path
import pandas as pd

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from statistics import mean
from math import log10
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
from transforms import *

class DIV2KDataset(Dataset):
    def __init__(self, hr_dir, transform=None, scale_factor=4, crop_size=256):
        self.hr_dir = hr_dir
        self.hr_images = sorted(os.listdir(hr_dir))
        self.transform = transform
        self.scale_factor = scale_factor
        self.crop_size = crop_size  # Fixed size for HR images

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        # Load high-resolution image
        hr_image = Image.open(os.path.join(self.hr_dir, self.hr_images[idx])).convert("RGB")

        # Resize high-resolution image to a fixed crop size
        hr_image = hr_image.resize((self.crop_size, self.crop_size), Image.BICUBIC)

        # Generate low-resolution image by downscaling
        lr_image = hr_image.resize(
            (self.crop_size // self.scale_factor, self.crop_size // self.scale_factor),
            Image.BICUBIC
        )

        # Apply transformations
        if self.transform:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)

        return lr_image, hr_image