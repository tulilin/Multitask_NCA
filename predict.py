# Step 3: predicting semantic segmentation and change detection results

import os
import torch
import torch.nn as nn
import numpy as np
from linear_transform import linear_transform
from Multitask_model import MultiTaskNet
from dataset_for_predict import Predict_Dataset
import numpy as np
import torch.nn as nn
from osgeo import gdal
from tqdm import tqdm
import math
from skimage import filters

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def writeTiff(im_data, im_width, im_height, im_bands, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, im_width, im_height, im_bands, datatype)
    if(dataset != None):
        dataset.SetGeoTransform(im_geotrans)
        dataset.SetProjection(im_proj)
    for i in range(im_bands):
        dataset.GetRasterBand(i+1).WriteArray(im_data[i])
    del dataset

def predict_semantic(model, dataset, classnum, device):
    overlap = dataset.grid - dataset.stride
    invalid_num = int(overlap / 2)

    h = int(dataset.height / dataset.stride) * (dataset.stride - 1) + dataset.grid
    w = int(dataset.width / dataset.stride) * (dataset.stride - 1) + dataset.grid

    feature_t1_semantic = np.zeros((classnum, h, w), dtype=np.float32)
    feature_t2_semantic = np.zeros((classnum, h, w), dtype=np.float32)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    model.eval()
    with torch.no_grad():
        for x, y, image_t1, image_t2 in tqdm(dataloader):
            image_t1 = image_t1.to(device)
            image_t2 = image_t2.to(device)

            feature_t1_semantic_patch, feature_t2_semantic_patch, _, _ = model(image_t1, image_t2)
            feature_t1_semantic_patch = torch.softmax(feature_t1_semantic_patch, dim=1)
            feature_t2_semantic_patch = torch.softmax(feature_t2_semantic_patch, dim=1)
            feature_t1_semantic_patch = feature_t1_semantic_patch.cpu().detach().numpy()
            feature_t2_semantic_patch = feature_t2_semantic_patch.cpu().detach().numpy()
            x = x[0].cpu().detach().numpy()
            y = y[0].cpu().detach().numpy()

            if((x == 0) | (y == 0)):
                feature_t1_semantic[:, x:x + dataset.grid - invalid_num, y:y + dataset.grid - invalid_num] = \
                    feature_t1_semantic_patch[0, :, 0:dataset.grid - invalid_num, 0:dataset.grid - invalid_num]

                feature_t2_semantic[:, x:x + dataset.grid - invalid_num, y:y + dataset.grid - invalid_num] = \
                    feature_t2_semantic_patch[0, :, 0:dataset.grid - invalid_num, 0:dataset.grid - invalid_num]
            else:
                feature_t1_semantic[:, x + invalid_num:x + dataset.grid - invalid_num, y + invalid_num:y + dataset.grid - invalid_num] = \
                    feature_t1_semantic_patch[0, :, invalid_num:dataset.grid - invalid_num, invalid_num:dataset.grid - invalid_num]

                feature_t2_semantic[:, x + invalid_num:x + dataset.grid - invalid_num, y + invalid_num:y + dataset.grid - invalid_num] = \
                    feature_t2_semantic_patch[0, :, invalid_num:dataset.grid - invalid_num, invalid_num:dataset.grid - invalid_num]

    feature_t1_semantic = feature_t1_semantic[:, 0:dataset.height, 0:dataset.width]
    feature_t2_semantic = feature_t2_semantic[:, 0:dataset.height, 0:dataset.width]

    return feature_t1_semantic, feature_t2_semantic

def predict_change(model, dataset, output_size, device):
    overlap = dataset.grid - dataset.stride
    invalid_num = int(overlap / 2)

    h = int(dataset.height / dataset.stride) * (dataset.stride - 1) + dataset.grid
    w = int(dataset.width / dataset.stride) * (dataset.stride - 1) + dataset.grid

    feature_t1_change = np.zeros((output_size, h, w), dtype=np.float32)
    feature_t2_change = np.zeros((output_size, h, w), dtype=np.float32)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    model.eval()
    with torch.no_grad():
        for x, y, image_t1, image_t2 in tqdm(dataloader):
            image_t1 = image_t1.to(device)
            image_t2 = image_t2.to(device)

            _, _, feature_t1_change_patch, feature_t2_change_patch = model(image_t1, image_t2)
            feature_t1_change_patch = feature_t1_change_patch.cpu().detach().numpy()
            feature_t2_change_patch = feature_t2_change_patch.cpu().detach().numpy()
            x = x[0].cpu().detach().numpy()
            y = y[0].cpu().detach().numpy()

            if((x == 0) | (y == 0)):
                feature_t1_change[:, x:x + dataset.grid - invalid_num, y:y + dataset.grid - invalid_num] = \
                    feature_t1_change_patch[0, :, 0:dataset.grid - invalid_num, 0:dataset.grid - invalid_num]

                feature_t2_change[:, x:x + dataset.grid - invalid_num, y:y + dataset.grid - invalid_num] = \
                    feature_t2_change_patch[0, :, 0:dataset.grid - invalid_num, 0:dataset.grid - invalid_num]
            else:
                feature_t1_change[:, x + invalid_num:x + dataset.grid - invalid_num, y + invalid_num:y + dataset.grid - invalid_num] = \
                    feature_t1_change_patch[0, :, invalid_num:dataset.grid - invalid_num, invalid_num:dataset.grid - invalid_num]

                feature_t2_change[:, x + invalid_num:x + dataset.grid - invalid_num, y + invalid_num:y + dataset.grid - invalid_num] = \
                    feature_t2_change_patch[0, :, invalid_num:dataset.grid - invalid_num, invalid_num:dataset.grid - invalid_num]

    feature_t1_change = feature_t1_change[:, 0:dataset.height, 0:dataset.width]
    feature_t2_change = feature_t2_change[:, 0:dataset.height, 0:dataset.width]

    return feature_t1_change, feature_t2_change

if __name__ == '__main__':
    print('Build model.')
    config = dict(
        in_channels=32,
        output_features=4,
        block_channels=(96, 128, 192, 256),
        num_blocks=(1, 1, 1, 1),
        inner_dim=128,
        reduction_ratio=1.0,
    )

    classnum = 4
    outdim_size = 10

    model = MultiTaskNet(config_backbone=config, input_size_seg=config['inner_dim'],
                         input_size_change=config['output_features'], output_size_change=outdim_size)
    model = torch.nn.DataParallel(model)

    model_path = './model/MT_qinzhou.pth'
    print('Current model:', model_path)
    model.load_state_dict(torch.load(model_path))
    print('Loaded trained model.')

    print('Load dataset.')
    t1_image_path = 'qinzhou_t1.tif'
    t1_dataset = gdal.Open(t1_image_path, gdal.GA_ReadOnly)
    width = t1_dataset.RasterXSize
    height = t1_dataset.RasterYSize
    gt = t1_dataset.GetGeoTransform()
    proj = t1_dataset.GetProjection()
    t1_mean = np.load('qinzhou_t1_mean.npy')
    t1_std = np.load('qinzhou_t1_std.npy')

    t2_image_path = 'qinzhou_t2.tif'
    t2_dataset = gdal.Open(t2_image_path, gdal.GA_ReadOnly)
    t2_mean = np.load('qinzhou_t2_mean.npy')
    t2_std = np.load('qinzhou_t2_std.npy')

    predict_dataset = Predict_Dataset(t1_image_path, t2_image_path, height, width, t1_mean, t1_std, t2_mean, t2_std,
                                      grid=256, stride=192)

    save_path = './result/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print('Predicting segmentation.')
    feature_t1_semantic, feature_t2_semantic = predict_semantic(model, predict_dataset, classnum, device)

    semantic_t1 = np.argmax(feature_t1_semantic, axis=0) + 1
    semantic_t1 = semantic_t1.astype(np.uint8)

    semantic_t2 = np.argmax(feature_t2_semantic, axis=0) + 1
    semantic_t2 = semantic_t2.astype(np.uint8)

    import gc
    del feature_t1_semantic, feature_t2_semantic
    gc.collect()

    writeTiff(semantic_t1, width, height, 1, gt, proj, save_path + 'semantic_t1.tif')
    writeTiff(semantic_t2, width, height, 1, gt, proj, save_path + 'semantic_t2.tif')

    print('Predicting change.')
    feature_t1_change, feature_t2_change = predict_change(model, predict_dataset, outdim_size, device)

    semantic_t1 = semantic_t1.reshape(-1)
    semantic_t2 = semantic_t2.reshape(-1)
    index = np.where(semantic_t1 == semantic_t2)[0]

    feature_t1 = feature_t1_change.reshape(feature_t1_change.shape[0], -1).transpose(1, 0)
    feature_t2 = feature_t2_change.reshape(feature_t2_change.shape[0], -1).transpose(1, 0)
    transform = linear_transform()
    transform.fit(feature_t1[index, :], feature_t2[index, :], outdim_size=outdim_size)
    feature_t1, feature_t2 = transform.test(feature_t1, feature_t2)

    feature_t1 = feature_t1.transpose(1, 0).reshape(outdim_size, height, width)
    feature_t2 = feature_t2.transpose(1, 0).reshape(outdim_size, height, width)

    feature_diff = feature_t1 - feature_t2
    feature_diff = np.reshape(feature_diff, [outdim_size, -1])
    std = np.std(feature_diff, axis=1, keepdims=True)
    feature_diff = feature_diff / std

    change_intensity = np.sum(feature_diff ** 2, axis=0, keepdims=True).reshape([1, height, width])
    change_intensity = np.squeeze(change_intensity)

    change_intensity_flatten = change_intensity.ravel()
    change_intensity_sorted = -np.sort(-change_intensity_flatten)
    threshold = change_intensity_sorted[int(len(change_intensity_sorted) * 0.005)]

    change_intensity[change_intensity >= threshold] = threshold

    change_intensity = (change_intensity - np.min(change_intensity)) / (
            np.max(change_intensity) - np.min(change_intensity))

    threshold_ostu = filters.threshold_otsu(change_intensity)
    change_map = (change_intensity >= threshold_ostu).astype(np.uint8)

    writeTiff(change_map, width, height, 1, gt, proj, save_path + 'change_map.tif')




