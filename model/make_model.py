import torch
import torch.nn as nn

from loss.CircleLoss import CircleSoftmax
from model.backbones.PyConv import pyconvhgresnet101
from model.backbones.densenet import densenet161
from model.backbones.res2net import res2net101_v1b
from model.models import resnet50_sw
from model.models.backbones.iresnet import resnet101_ir
from model.models.backbones.resnest import resnest50, resnest101, resnest200, resnest269, resnest269_sw
from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from loss.arcface import ArcFace
from .backbones.resnet_ibn_a import resnet50_ibn_a,resnet101_ibn_a
from .backbones.se_resnet_ibn_a import se_resnet101_ibn_a
import torch.nn.functional as F
class GeM(nn.Module):

    def __init__(self, p=3.0, eps=1e-6, freeze_p=True):
        super(GeM, self).__init__()
        self.p = p if freeze_p else Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p),
                            (1, 1)).pow(1. / self.p)

    def __repr__(self):
        if isinstance(self.p, float):
            p = self.p
        else:
            p = self.p.data.tolist()[0]
        return self.__class__.__name__ +\
               '(' + 'p=' + '{:.4f}'.format(p) +\
               ', ' + 'eps=' + str(self.eps) + ')'
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.efn=False
        self.dense=False
        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck, frozen_stages=cfg.MODEL.FROZEN,
                               layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')
        elif model_name == 'pyconv':
            self.in_planes = 2048
            self.base = pyconvhgresnet101()
        elif model_name == 'densenet':
            self.in_planes = 2208
            self.base = densenet161(pretrained=True)
            self.dense=True
        elif model_name =='res2net':
            self.in_planes = 2048
            self.base = res2net101_v1b(last_stride)
        elif model_name == 'resnest50':
            self.in_planes = 2048
            self.base = resnest50(last_stride)
            print('using resnest50 as a backbone')
        elif model_name == 'resnest101':
            self.in_planes = 2048
            self.base = resnest101(last_stride)
            print('using resnest101 as a backbone')
        elif model_name == 'resnest200':
            self.in_planes = 2048
            self.base = resnest200(last_stride)
            print('using resnest200 as a backbone')
        elif model_name == 'resnest269':
            self.in_planes = 2048
            self.base = resnest269(last_stride)
            print('using resnest269 as a backbone')
        elif model_name == 'efn':
            from efficientnet_pytorch import EfficientNet
            self.base = EfficientNet.from_pretrained('efficientnet-b7')
            self.in_planes=2560
            self.efn=True
        elif model_name == 'resnet_ir':
            self.in_planes = 512
            self.base = resnet101_ir(None)
            print('using resnet50 as a backbone')
        elif model_name == 'resnext':
            self.in_planes = 2048
            model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x16d_wsl')
            self.base = nn.Sequential(*list(model.children())[:8])
            print('using resnext as a backbone')
        elif model_name == 'resnet50_ibn_a':
            self.in_planes = 2048
            self.base = resnet50_ibn_a(last_stride)
            print('using resnet50_ibn_a as a backbone')
        elif model_name == 'resnet50_sw':
            self.in_planes = 2048
            self.base = resnet50_sw()
            print('using resnet50_ibn_a as a backbone')
        elif model_name == 'resnet101_ibn_a':
            self.in_planes = 2048
            self.base = resnet101_ibn_a(last_stride, frozen_stages=cfg.MODEL.FROZEN)
            print('using resnet101_ibn_a as a backbone')
        elif model_name == 'se_resnet101_ibn_a':
            self.in_planes = 2048
            self.base = se_resnet101_ibn_a(last_stride,frozen_stages=cfg.MODEL.FROZEN)
            print('using se_resnet101_ibn_a as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        # self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = G()
        self.num_classes = num_classes

        if self.cos_layer:
            print('using cosine layer')
            self.arcface = CircleSoftmax(self.in_planes, self.num_classes, s=30.0, m=0.50)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None):  # label is unused if self.cos_layer == 'no'
        if self.efn is True:
            x = self.base.extract_features(x)
        else:
            x = self.base(x)
        if self.dense is True:
            global_feat = x
            feat = self.bottleneck(global_feat)
        else:
            global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
            global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
            feat = self.bottleneck(global_feat)


        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i or 'arcface' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

def make_model(cfg, num_class):
    model = Backbone(num_class, cfg)
    return model
