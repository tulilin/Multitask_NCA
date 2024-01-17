import torch
import torch.nn as nn
import numpy as np
from SAHR_Net import SAHR_Net

class ConvNet_seg(nn.Module):
    def __init__(self, input_size, output_size):
        super(ConvNet_seg, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.conv1 = nn.Conv2d(in_channels=self.input_size, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv_final = nn.Conv2d(in_channels=64, out_channels=self.output_size, kernel_size=1)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.activation(x1)
        x1 = self.bn1(x1)

        x2 = self.conv2(x1)
        x2 = self.activation(x2)
        x2 = self.bn2(x2)

        x3 = self.conv3(x2)
        x3 = self.activation(x3)
        x3 = self.bn3(x3)

        output = self.conv_final(x3)

        return output

class ConvNet_change(nn.Module):
    def __init__(self, input_size, output_size):
        super(ConvNet_change, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.conv1 = nn.Conv2d(in_channels=self.input_size, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv_final = nn.Conv2d(in_channels=64, out_channels=self.output_size, kernel_size=1)

        self.activation = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.activation(x1)
        x1 = self.bn1(x1)

        x2 = self.conv2(x1)
        x2 = self.activation(x2)
        x2 = self.bn2(x2)

        x3 = self.conv3(x2)
        x3 = self.activation(x3)
        x3 = self.bn3(x3)

        output = self.conv_final(x3)

        return output

class MultiTaskNet(nn.Module):
    def __init__(self, config_backbone, input_size_seg, input_size_change, output_size_change):
        super(MultiTaskNet, self).__init__()
        self.config = config_backbone
        self.input_size_seg = input_size_seg
        self.input_size_change = input_size_change
        self.output_size_seg = self.config['output_features']
        self.output_size_change = output_size_change

        # Feature extractor
        self.backbone = SAHR_Net(config=self.config)

        # Semantic Segmentation module
        self.seg_model = ConvNet_seg(input_size=self.input_size_seg, output_size=self.output_size_seg)

        # Change Detection module
        self.change_model1 = ConvNet_change(input_size=self.input_size_change, output_size=self.output_size_change)
        self.change_model2 = ConvNet_change(input_size=self.input_size_change, output_size=self.output_size_change)

    def forward(self, x1, x2):
        feat1, feat2, logits1, logits2 = self.backbone(x1, x2)

        output1_seg = self.seg_model(feat1)
        output2_seg = self.seg_model(feat2)

        output1_change = self.change_model1(logits1)
        output2_change = self.change_model2(logits2)

        return output1_seg, output2_seg, output1_change, output2_change