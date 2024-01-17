import os
import torch
from torch.utils import data
import numpy as np
from osgeo import gdal
import random

class OHS_Dataset_patch_bitemporal(data.Dataset):
    def __init__(self, image_file_list_t1, image_file_list_t2, label_file_list, crop_size=256, overlap=64):
        self.image_file_list_t1 = image_file_list_t1
        self.image_file_list_t2 = image_file_list_t2
        self.label_file_list = label_file_list
        self.crop_size = crop_size
        self.overlap = overlap

    def __len__(self):
        return len(self.image_file_list_t1)

    def __getitem__(self, index):
        image_file_t1 = self.image_file_list_t1[index]
        image_file_t2 = self.image_file_list_t2[index]
        label_file = self.label_file_list[index]

        image_dataset_t1 = gdal.Open(image_file_t1, gdal.GA_ReadOnly)
        image_dataset_t2 = gdal.Open(image_file_t2, gdal.GA_ReadOnly)
        label_dataset = gdal.Open(label_file, gdal.GA_ReadOnly)

        image_t1 = image_dataset_t1.ReadAsArray()
        image_t2 = image_dataset_t2.ReadAsArray()
        label = label_dataset.ReadAsArray()

        image_t1 = torch.tensor(image_t1, dtype=torch.float)
        image_t2 = torch.tensor(image_t2, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long) - 1

        return image_t1, image_t2, label

class MultiTask_Dataset(data.Dataset):
    def __init__(self, image_file_list_t1, image_file_list_t2):
        self.image_file_list_t1 = image_file_list_t1
        self.image_file_list_t2 = image_file_list_t2

        self.image_file_list_t2_shuffle = self.image_file_list_t2.copy()
        random.shuffle(self.image_file_list_t2_shuffle)

    def shuffle(self):
        random.shuffle(self.image_file_list_t2_shuffle)

    def __len__(self):
        return len(self.image_file_list_t1)

    def __getitem__(self, index):
        image_file_t1 = self.image_file_list_t1[index]
        image_file_t2 = self.image_file_list_t2[index]
        image_file_t2_shuffle = self.image_file_list_t2_shuffle[index]

        label_file = image_file_t1.replace('t1', 'label')
        label_file_shuffle = image_file_t2_shuffle.replace('t2', 'label')

        image_dataset_t1 = gdal.Open(image_file_t1, gdal.GA_ReadOnly)
        image_dataset_t2 = gdal.Open(image_file_t2, gdal.GA_ReadOnly)
        image_dataset_t2_shuffle = gdal.Open(image_file_t2_shuffle, gdal.GA_ReadOnly)

        label_dataset = gdal.Open(label_file, gdal.GA_ReadOnly)
        label_dataset_shuffle = gdal.Open(label_file_shuffle, gdal.GA_ReadOnly)

        image_t1 = image_dataset_t1.ReadAsArray()
        image_t2 = image_dataset_t2.ReadAsArray()
        image_t2_shuffle = image_dataset_t2_shuffle.ReadAsArray()

        label = label_dataset.ReadAsArray()
        label_shuffle = label_dataset_shuffle.ReadAsArray()

        image_t1 = torch.tensor(image_t1, dtype=torch.float)
        image_t2 = torch.tensor(image_t2, dtype=torch.float)
        image_t2_shuffle = torch.tensor(image_t2_shuffle, dtype=torch.float)

        label = torch.tensor(label, dtype=torch.long) - 1
        label_shuffle = torch.tensor(label_shuffle, dtype=torch.long) - 1

        return image_t1, image_t2, image_t2_shuffle, label, label_shuffle