# Step 1: training the SAHR_Net feature extractor

import torch
import os
from dataset import OHS_Dataset_patch_bitemporal
from SAHR_Net import SAHR_Net
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tqdm import tqdm
from osgeo import gdal
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP','.tif'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

class_num = 4
epoch_num = 100
batch_size = 4

config = dict(
        in_channels=32,
        output_features=4,
        block_channels=(96, 128, 192, 256),
        num_blocks=(1, 1, 1, 1),
        inner_dim=128,
        reduction_ratio=1.0,
    )

def main():
    print('Build model ...')
    model = SAHR_Net(config).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0001)

    lambda_lr = lambda x: (1 - x / epoch_num) ** 0.9
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

    weight = [2.8358, 0.2989, 4.6014, 11.9034]
    weight = torch.tensor(weight, dtype=torch.float).to(device)

    criterion_seg = torch.nn.CrossEntropyLoss(ignore_index=-1, weight=weight)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    data_dir1 = './train/t1/'
    data_dir2 = './train/t2/'
    label_dir = './train/label/'

    image_list_t1 = []
    image_list_t2 = []
    label_list = []

    for root, paths, fnames in sorted(os.walk(data_dir1)):
        for fname in fnames:
            if is_image_file(fname):
                image_path_t1 = os.path.join(data_dir1, fname)
                image_path_t2 = os.path.join(data_dir2, fname)
                label_path = os.path.join(label_dir, fname)
                assert os.path.exists(image_path_t1)
                assert os.path.exists(image_path_t2)
                assert os.path.exists(label_path)
                image_list_t1.append(image_path_t1)
                image_list_t2.append(image_path_t2)
                label_list.append(label_path)

    assert len(image_list_t1) == len(label_list)
    assert len(image_list_t2) == len(label_list)

    # 构建训练和验证数据集
    train_dataset = OHS_Dataset_patch_bitemporal(image_file_list_t1=image_list_t1, image_file_list_t2=image_list_t2,
                                                 label_file_list=label_list)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2,
                                               pin_memory=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    model_path = './model/SAHR_Net/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    for epoch in range(epoch_num):
        print('Epoch: %d/%d' % (epoch + 1, epoch_num))
        print('Current learning rate: %.8f' % (optimizer.state_dict()['param_groups'][0]['lr']))
        model.train()
        batch_index = 0
        loss_sum = 0
        for data_t1, data_t2, label in tqdm(train_loader):
            data_t1 = data_t1.to(device)
            data_t2 = data_t2.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            _, _, feature_t1, feature_t2 = model(data_t1, data_t2)

            loss_seg_t1 = criterion_seg(feature_t1, label)
            loss_seg_t2 = criterion_seg(feature_t2, label)

            loss = 0.5 * loss_seg_t1 + 0.5 * loss_seg_t2

            loss.backward()
            optimizer.step()

            loss_sum = loss_sum + loss.item()
            batch_index = batch_index + 1
            average_loss_cur = loss_sum / batch_index
            if(batch_index % 5 == 0):
                print('Average Training Loss %.6f, Current: Seg Loss t1 %.6f, Seg Loss t2 %.6f'
                      % (average_loss_cur, loss_seg_t1.item(), loss_seg_t2.item()))

        average_loss = loss_sum / batch_index
        print('Epoch [%d/%d] training loss %.6f' % (epoch + 1, epoch_num, average_loss))

        if(epoch % 10 == 0):
            torch.save(model.state_dict(), model_path + 'SAHR_Net_' + str(epoch) + '.pth')

        scheduler.step()

    torch.save(model.state_dict(), model_path + 'SAHR_Net_final.pth')

if __name__ == '__main__':
    main()

