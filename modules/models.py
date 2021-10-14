import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

#################### Not Used ##########################


def init_weights(m):  # model.apply(init_weights)
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal(m.weight)
    elif isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class SEBlock(nn.Module):
    def __init__(self, in_channel, reduction_rate):
        super(SEBlock, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Linear(in_channel, in_channel//reduction_rate)
        # ReLU
        self.linear2 = nn.Linear(in_channel//reduction_rate, in_channel)

    def forward(self, x):
        '''
        |x| = (B, T, H, W)
        |out| = (B, T, 1, 1)
        '''
        out = self.gap(x)
        out = torch.squeeze(out)
        out = F.relu(self.linear1(out), inplace=True)
        out = self.linear2(out)
        return F.sigmoid(out)


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, reduction_rate):
        super(BasicConv, self).__init__()
        self.conv_1 = nn.Conv2d(in_channel, in_channel //
                                reduction_rate, kernel_size=3, padding=1)
        self.batchnorm_1 = nn.BatchNorm2d(in_channel)
        self.conv_2 = nn.Conv2d(
            in_channel//reduction_rate, out_channel, kernel_size=3, padding=1)
        self.batchnorm_2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        out = self.conv_1(x)
        out = F.relu(self.batchnorm_1(out), inplace=True)
        out = self.conv_2(out)
        out = F.relu(self.batchnorm_2(out), inplace=True)
        return out


class BasicConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, reduction_rate):
        super(BasicConvBlock, self).__init__()
        self.basicconv = BasicConv(
            in_channel, out_channel, reduction_rate=reduction_rate)
        self.se = SEBlock(out_channel, reduction_rate)
        if in_channel == out_channel:
            self.projection = None
        else:
            self.projection = nn.Conv2d(in_channel, out_channel, kernel_size=1)

    def forward(self, x):
        out = self.basicconv(x)
        se_branch = self.se(out)
        se_branch = torch.unsqueeze(se_branch, dim=2)
        se_branch = torch.unsqueeze(se_branch, dim=3)
        out = torch.mul(out, se_branch)
        if self.projection:
            x = self.projection(x)
        return x + out
##############################################################


class EffnetV2(nn.Module):
    def __init__(self, weight_name, num_classes):
        super(EffnetV2, self).__init__()
        # tf_efficientnetv2_m_in21ft1k
        self.effnet = timm.create_model(
            weight_name, pretrained=True, num_classes=num_classes)
        # EfficientNetV2M 21K Pretrained, 1K Fine-Tuned

    def forward(self, x):
        out = self.effnet(x)
        return out


class Effnet(nn.Module):
    def __init__(self, weight_name, num_classes):
        super(Effnet, self).__init__()
        # tf_efficientnet_b7_ns
        self.effnet = timm.create_model(
            weight_name, pretrained=True, num_classes=num_classes)
        head = nn.Sequential(self.effnet.conv_stem,
                             self.effnet.bn1, self.effnet.act1)
        blocks = list(self.effnet.blocks.children())
        tail = nn.Sequential(self.effnet.conv_head, self.effnet.bn2,
                             self.effnet.act2, self.effnet.global_pool)
        classifier = nn.Sequential(nn.Dropout(
            p=0.5, inplace=False), self.effnet.classifier)
        blocks.insert(0, head)
        blocks.append(tail)
        blocks.append(classifier)
        self.effnet = nn.Sequential(*blocks)

    def forward(self, x):
        out = self.effnet(x)
        return out


class Swin(nn.Module):
    def __init__(self, weight_name, num_classes):
        super(Swin, self).__init__()
        # swin_large_patch4_window12_384
        self.swin = timm.create_model(
            weight_name, pretrained=True, num_classes=num_classes)

    def forward(self, x):
        out = self.swin(x)
        return out


def get_model(cfg, num_classes):
    if cfg['MODEL']['MODEL_NAME'] == 'EffNet':
        return Effnet(weight_name=cfg['MODEL']['WEIGHT_NAME'], num_classes=num_classes)
    elif cfg['MODEL']['MODEL_NAME'] == 'Swin':
        return Swin(weight_name=cfg['MODEL']['WEIGHT_NAME'], num_classes=num_classes)
    elif cfg['MODEL']['MODEL_NAME'] == 'EffNetV2':
        return EffnetV2(weight_name=cfg['MODEL']['WEIGHT_NAME'], num_classes=num_classes)
