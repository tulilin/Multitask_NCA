# Step 4: generating NCA according to semantic segmentation and change detection results

import os
import numpy as np
from osgeo import gdal

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

change_map_path = './result/change_map.tif'

segment_map_t1_path = './result/semantic_t1.tif'
segment_map_t2_path = './result/semantic_t2.tif'

save_path = './result/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

segment_map_t1_dataset = gdal.Open(segment_map_t1_path, gdal.GA_ReadOnly)
width = segment_map_t1_dataset.RasterXSize
height = segment_map_t1_dataset.RasterYSize
gt = segment_map_t1_dataset.GetGeoTransform()
proj = segment_map_t1_dataset.GetProjection()
segment_map_t1 = segment_map_t1_dataset.ReadAsArray()

segment_map_t2_dataset = gdal.Open(segment_map_t2_path, gdal.GA_ReadOnly)
segment_map_t2 = segment_map_t2_dataset.ReadAsArray()
segment_map_t2 = segment_map_t2[0:height, 0:width]

change_map_dataset = gdal.Open(change_map_path, gdal.GA_ReadOnly)
change_map = change_map_dataset.ReadAsArray()
change_map = change_map[0:height, 0:width]

change_map_post = change_map.copy()
change_map_post[segment_map_t1 == segment_map_t2] = 0

new_constructed_map = change_map_post.copy()
new_constructed_map[segment_map_t2 != 4] = 0

writeTiff(new_constructed_map, width, height, 1, gt, proj, save_path + 'NCA.tif')
