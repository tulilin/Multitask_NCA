# Step 2: training the semantic segmentation module and DMAD change detection module in the multi-task framework

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from Multitask_model import MultiTaskNet
from dataset import MultiTask_Dataset
from objectives import loss_func
import time
import numpy as np
from osgeo import gdal
from tqdm import tqdm
from hist_match import hist_match

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP','.tif'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

class Solver():
    def __init__(self, model, outdim_size, epoch_num, batch_size, weight, learning_rate, reg_par, device=torch.device('cpu'), start_epoch=0):
        self.model = nn.DataParallel(model)
        self.model.to(device)
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.start_epoch = start_epoch

        self.seg_loss = torch.nn.CrossEntropyLoss(ignore_index=-1, weight=weight)
        self.change_loss = loss_func(outdim_size, use_all_singular_values=False, device='cuda').loss
        self.contrastive_loss = torch.nn.MSELoss()

        self.trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
        self.optimizer = torch.optim.Adam(
            self.trainable_params, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=reg_par)

        lambda_lr = lambda x: (1 - x / epoch_num) ** 0.9
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_lr, last_epoch=self.start_epoch-1)

        self.device = device

        self.outdim_size = outdim_size

    def fit(self, train_loader, model_path):
        for epoch in range(self.start_epoch, self.epoch_num):
            print('Epoch: %d/%d' % (epoch + 1, self.epoch_num))
            print('Current learning rate: %.8f' % (self.optimizer.state_dict()['param_groups'][0]['lr']))
            train_loader.dataset.shuffle()
            batch_index = 0
            loss_sum = 0
            self.model.train()

            for data_t1, data_t2, data_t2_shuffle, label, label_shuffle in tqdm(train_loader):
                data_t1 = data_t1.to(self.device)
                data_t2 = data_t2.to(self.device)
                data_t2_shuffle = data_t2_shuffle.to(self.device)
                label = label.to(self.device)
                label_shuffle = label_shuffle.to(self.device)

                o1_seg, o2_seg, o1_change, o2_change = self.model(data_t1, data_t2)
                o1_seg, o2_shuffle_seg, o1_change, o2_shuffle_change = self.model(data_t1, data_t2_shuffle)

                batch_size_cur = o1_change.size(0)
                patch_size = o1_change.size(-1)

                o1_change = o1_change.permute(0, 2, 3, 1).reshape(-1, self.outdim_size)
                o2_change = o2_change.permute(0, 2, 3, 1).reshape(-1, self.outdim_size)
                o2_shuffle_change = o2_shuffle_change.permute(0, 2, 3, 1).reshape(-1, self.outdim_size)

                loss_seg_t1 = self.seg_loss(o1_seg, label)
                loss_seg_t2 = self.seg_loss(o2_seg, label)
                loss_seg_t2_shuffle = self.seg_loss(o2_shuffle_seg, label_shuffle)

                loss_seg = (loss_seg_t1 + loss_seg_t2 + loss_seg_t2_shuffle) / 3.0

                loss_change_1, _, _ = self.change_loss(o1_change, o2_shuffle_change)

                o1_seg_prob = torch.softmax(o1_seg, dim=1)
                o2_seg_prob = torch.softmax(o2_seg, dim=1)

                o1_seg_result = torch.argmax(o1_seg, dim=1)
                o2_seg_result = torch.argmax(o2_seg, dim=1)

                semantic_diff = o1_seg_prob - o2_seg_prob
                semantic_diff = torch.sum(semantic_diff ** 2, dim=1)

                o1_seg_result = o1_seg_result.reshape(-1)
                o2_seg_result = o2_seg_result.reshape(-1)

                index = torch.where(o1_seg_result == o2_seg_result)[0]

                o1_change_2 = o1_change[index, :]
                o2_change_2 = o2_change[index, :]

                loss_change_2_raw, M, w = self.change_loss(o1_change_2, o2_change_2)
                loss_change_2 = torch.trace(torch.eye(self.outdim_size)) - loss_change_2_raw

                loss_change = 0.5 * loss_change_1 + 0.5 * loss_change_2

                o1_output = torch.matmul(o1_change - M[0].reshape(1, -1), w[0]).reshape(batch_size_cur, patch_size, patch_size, self.outdim_size)
                o2_output = torch.matmul(o2_change - M[1].reshape(1, -1), w[1]).reshape(batch_size_cur, patch_size, patch_size, self.outdim_size)
                o1_output = o1_output.permute(0, 3, 1, 2)
                o2_output = o2_output.permute(0, 3, 1, 2)

                change_diff = o1_output - o2_output
                change_diff = change_diff.reshape(batch_size_cur, self.outdim_size, -1)
                std = torch.std(change_diff, dim=2, keepdim=True)
                change_diff = change_diff / std
                change_diff = torch.sum(change_diff ** 2, dim=1)
                change_diff = change_diff.reshape(batch_size_cur, patch_size, patch_size)

                change_diff_match = torch.zeros_like(change_diff).to(self.device)
                for i in range(batch_size_cur):
                    change_diff_cur = torch.floor(change_diff[i, :, :])
                    semantic_diff_cur = torch.floor(semantic_diff[i, :, :] * 128.0)
                    change_diff_match_cur = hist_match(change_diff_cur, semantic_diff_cur)
                    change_diff_match_cur = change_diff_match_cur / 128.0
                    change_diff_match[i, :, :] = change_diff_match_cur

                loss_consist = self.contrastive_loss(semantic_diff, change_diff_match)

                lambda_change = 0.1
                lambda_consist = 10
                loss = loss_seg + lambda_change * loss_change + lambda_consist * loss_consist

                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()

                loss_sum = loss_sum + loss.item()

                batch_index = batch_index + 1
                average_loss_cur = loss_sum / batch_index

                if (batch_index % 5 == 0):
                    print('Loss: %.6f' % (average_loss_cur))

            torch.save(self.model.state_dict(), model_path + 'Multitask_' + str(epoch) + '.pth')

            self.scheduler.step()

        torch.save(self.model.state_dict(), model_path + 'Multitask_final.pth')

def main():
    config = dict(
        in_channels=32,
        output_features=4,
        block_channels=(96, 128, 192, 256),
        num_blocks=(1, 1, 1, 1),
        inner_dim=128,
        reduction_ratio=1.0,
    )

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print("Using", torch.cuda.device_count(), "GPUs")

    outdim_size = 10

    learning_rate = 1e-4
    epoch_num = 50
    batch_size = 4

    reg_par = 1e-5

    data_dir_t1 = './train/t1/'
    data_dir_t2 = './train/t2/'

    image_list_t1 = []
    image_list_t2 = []

    for root, paths, fnames in sorted(os.walk(data_dir_t1)):
        for fname in fnames:
            if is_image_file(fname):
                image_path_t1 = os.path.join(data_dir_t1, fname)
                image_path_t2 = os.path.join(data_dir_t2, fname)
                assert os.path.exists(image_path_t1)
                assert os.path.exists(image_path_t2)
                image_list_t1.append(image_path_t1)
                image_list_t2.append(image_path_t2)

    assert len(image_list_t1) == len(image_list_t2)

    train_dataset = MultiTask_Dataset(image_list_t1, image_list_t2)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2,
                                               pin_memory=True)

    weight = [2.8358, 0.2989, 4.6014, 11.9034]
    weight = torch.tensor(weight, dtype=torch.float).to(device)

    model = MultiTaskNet(config_backbone=config, input_size_seg=config['inner_dim'], input_size_change=config['output_features'], output_size_change=outdim_size)

    pretrain_path = './model/SAHR_Net/SAHR_Net_final.pth'
    model.backbone.load_state_dict(torch.load(pretrain_path))
    print('Loaded pretrained model for backbone.')
    start_epoch = 0

    for param in model.backbone.parameters():
        param.requires_grad = False

    solver = Solver(model, outdim_size, epoch_num, batch_size, weight, learning_rate, reg_par, device=device, start_epoch=start_epoch)

    model_path = './model/Multitask/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    solver.fit(train_loader, model_path)

if __name__ == '__main__':
    main()