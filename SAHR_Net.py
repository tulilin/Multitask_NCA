# SAHR-Net feature extractor
# Modified from FreeNet:
# https://github.com/Z-Zheng/FreeNet/blob/master/module/freenet.py

import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio):
        super(SEBlock, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.seq = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        v = self.gap(x)
        score = self.seq(v.view(v.size(0), v.size(1)))
        y = x * score.view(score.size(0), score.size(1), 1, 1)
        return y

def conv1x1_bn_relu(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True),
    )

def conv3x3_bn_relu(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 3, 1, 1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True),
    )

def repeat_block(block_channel, r, n):
    layers = [
        nn.Sequential(
            SEBlock(block_channel, r),
            conv3x3_bn_relu(block_channel, block_channel)
        )
        for _ in range(n)]
    return nn.Sequential(*layers)

class SAHR_Net(nn.Module):
    def __init__(self, config):
        super(SAHR_Net, self).__init__()
        self.config = config
        r = int(16 * self.config['reduction_ratio'])
        block1_channels = int(self.config['block_channels'][0] * self.config['reduction_ratio'] / r) * r
        block2_channels = int(self.config['block_channels'][1] * self.config['reduction_ratio'] / r) * r
        block3_channels = int(self.config['block_channels'][2] * self.config['reduction_ratio'] / r) * r
        block4_channels = int(self.config['block_channels'][3] * self.config['reduction_ratio'] / r) * r

        self.feature_ops = nn.ModuleList([
            conv3x3_bn_relu(self.config['in_channels'], block1_channels),

            repeat_block(block1_channels, r, self.config['num_blocks'][0]),
            nn.Identity(),
            conv3x3_bn_relu(block1_channels, block2_channels),

            repeat_block(block2_channels, r, self.config['num_blocks'][1]),
            nn.Identity(),
            conv3x3_bn_relu(block2_channels, block3_channels),

            repeat_block(block3_channels, r, self.config['num_blocks'][2]),
            nn.Identity(),
            conv3x3_bn_relu(block3_channels, block4_channels),

            repeat_block(block4_channels, r, self.config['num_blocks'][3]),
            nn.Identity(),
        ])
        inner_dim = int(self.config['inner_dim'] * self.config['reduction_ratio'])

        self.reduce_1x1convs = nn.ModuleList([
            conv1x1_bn_relu(block1_channels, inner_dim),
            conv1x1_bn_relu(block2_channels, inner_dim),
            conv1x1_bn_relu(block3_channels, inner_dim),
            conv1x1_bn_relu(block4_channels, inner_dim),
        ])
        self.fuse_3x3convs = nn.ModuleList([
            conv3x3_bn_relu(inner_dim, inner_dim),
            conv3x3_bn_relu(inner_dim, inner_dim),
            conv3x3_bn_relu(inner_dim, inner_dim),
            conv3x3_bn_relu(inner_dim, inner_dim),
        ])

        self.cls_pred_conv = nn.Conv2d(inner_dim, self.config['output_features'], 1)

    def top_down(self, top, lateral):
        return lateral + top

    def forward_single(self, x):
        feat_list = []
        for op in self.feature_ops:
            x = op(x)
            if isinstance(op, nn.Identity):
                feat_list.append(x)

        inner_feat_list = [self.reduce_1x1convs[i](feat) for i, feat in enumerate(feat_list)]
        inner_feat_list.reverse()

        out_feat_list = [self.fuse_3x3convs[0](inner_feat_list[0])]
        for i in range(len(inner_feat_list) - 1):
            inner = self.top_down(out_feat_list[i], inner_feat_list[i + 1])
            out = self.fuse_3x3convs[i + 1](inner)
            out_feat_list.append(out)

        final_feat = out_feat_list[-1]

        logit = self.cls_pred_conv(final_feat)

        return final_feat, logit

    def forward(self, x_t1, x_t2=None):
        if (x_t2 is not None):
            final_feat_t1, logit_t1 = self.forward_single(x_t1)
            final_feat_t2, logit_t2 = self.forward_single(x_t2)

            return final_feat_t1, final_feat_t2, logit_t1, logit_t2

        else:
            final_feat, logit = self.forward_single(x_t1)
            return final_feat, logit