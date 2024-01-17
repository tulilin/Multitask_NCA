import os
import torch
from torch.utils import data
import numpy as np
from osgeo import gdal
from tqdm import tqdm

class Predict_Dataset(data.Dataset):
    def __init__(self, image_path_t1, image_path_t2, height, width, mean1, std1, mean2, std2, grid=256, stride=224):
        self.image_path_t1 = image_path_t1
        self.image_path_t2 = image_path_t2
        self.height = height
        self.width = width
        self.mean1 = mean1
        self.std1 = std1
        self.mean2 = mean2
        self.std2 = std2
        self.grid = grid
        self.stride = stride

        self.coordinate = []
        for x in range(0, self.height, self.stride):
            for y in range(0, self.width, self.stride):
                self.coordinate.append((x, y))

        self.num_patches = len(self.coordinate)

    def __len__(self):
        return self.num_patches

    def __getitem__(self, idx):
        dataset_t1 = gdal.Open(self.image_path_t1, gdal.GA_ReadOnly)
        dataset_t2 = gdal.Open(self.image_path_t2, gdal.GA_ReadOnly)

        x, y = self.coordinate[idx]
        dx = self.height - x
        dy = self.width - y
        if ((dx >= self.grid) & (dy >= self.grid)):
            image_patch_t1 = dataset_t1.ReadAsArray(y, x, self.grid, self.grid)
            image_patch_t2 = dataset_t2.ReadAsArray(y, x, self.grid, self.grid)
        elif ((dx >= self.grid) & (dy < self.grid)):
            image_patch_t1 = dataset_t1.ReadAsArray(y, x, dy, self.grid)
            image_patch_t2 = dataset_t2.ReadAsArray(y, x, dy, self.grid)
            image_patch_t1 = np.pad(image_patch_t1, ((0, 0), (0, 0), (0, self.grid - dy)))
            image_patch_t2 = np.pad(image_patch_t2, ((0, 0), (0, 0), (0, self.grid - dy)))
        elif ((dx < self.grid) & (dy >= self.grid)):
            image_patch_t1 = dataset_t1.ReadAsArray(y, x, self.grid, dx)
            image_patch_t2 = dataset_t2.ReadAsArray(y, x, self.grid, dx)
            image_patch_t1 = np.pad(image_patch_t1, ((0, 0), (0, self.grid - dx), (0, 0)))
            image_patch_t2 = np.pad(image_patch_t2, ((0, 0), (0, self.grid - dx), (0, 0)))
        elif ((dx < self.grid) & (dy < self.grid)):
            image_patch_t1 = dataset_t1.ReadAsArray(y, x, dy, dx)
            image_patch_t2 = dataset_t2.ReadAsArray(y, x, dy, dx)
            image_patch_t1 = np.pad(image_patch_t1, ((0, 0), (0, self.grid - dx), (0, self.grid - dy)))
            image_patch_t2 = np.pad(image_patch_t2, ((0, 0), (0, self.grid - dx), (0, self.grid - dy)))

        bands = image_patch_t1.shape[0]

        image_patch_t1 = image_patch_t1.transpose(1, 2, 0)
        image_patch_t2 = image_patch_t2.transpose(1, 2, 0)

        image_patch_t1 = np.reshape(image_patch_t1, newshape=[-1, bands])
        image_patch_t1 = image_patch_t1.reshape(-1, bands)
        image_patch_t1 = (image_patch_t1 - self.mean1) / self.std1
        image_patch_t1 = np.reshape(image_patch_t1, newshape=[self.grid, self.grid, bands]).transpose(2, 0, 1)
        image_patch_t1 = torch.tensor(image_patch_t1, dtype=torch.float)


        image_patch_t2 = np.reshape(image_patch_t2, newshape=[-1, bands])
        image_patch_t2 = image_patch_t2.reshape(-1, bands)
        image_patch_t2 = (image_patch_t2 - self.mean2) / self.std2
        image_patch_t2 = np.reshape(image_patch_t2, newshape=[self.grid, self.grid, bands]).transpose(2, 0, 1)
        image_patch_t2 = torch.tensor(image_patch_t2, dtype=torch.float)

        return x, y, image_patch_t1, image_patch_t2
