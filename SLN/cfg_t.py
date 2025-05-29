import torch
from easydict import EasyDict as E
from monai.transforms import (
    Compose, NormalizeIntensityd, SpatialPadd,
    EnsureTyped
)
from monai.transforms import (
    RandScaleIntensityd, 
    RandAdjustContrastd, 
    RandHistogramShiftd, 
    RandShiftIntensityd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandGaussianSharpend,
)
from transforms import LoadPickle, CombineSeq
from adan import Adan
from torch import nn

_ = torch.device("cuda")

C = E()
C.mode = 'val'
C.csv = './sample_data/dataset.csv'
C.workdir = './ckpts/c'
C.log = 'log_xx'
C.seg = 'seg_crop'
C.bz = 24
C.eps = 300
C.sz_t = [100,100,50]
C.lr = 1e-4
C.opt = Adan
C.loss = nn.CrossEntropyLoss()
C.ckpt = ''
C.excl = []

def _t(z):
    if z == 'train':
        return Compose([
            LoadPickle(keys=['tumor'], add_seg=C.seg),
            SpatialPadd(keys=['tumor'], spatial_size=C.sz_t),
            OneOf([ 
                RandScaleIntensityd(keys=["tumor"],prob=0.2,factors=0.4), 
                RandAdjustContrastd(keys=["tumor"],prob=0.2), 
                RandHistogramShiftd(keys=["tumor"],prob=0.2), 
                RandShiftIntensityd(keys=["tumor"],prob=0.2,offsets=0.2),
                RandGaussianNoised(keys=["tumor"], prob=0.2),
                RandGaussianSmoothd(keys=["tumor"], prob=0.2),
                RandGaussianSharpend(keys=["tumor"], prob=0.2),
            ]),
            NormalizeIntensityd(keys=['tumor'], nonzero=False, channel_wise=True),
            EnsureTyped(keys=['tumor', 'gt'], dtype=torch.float),
        ])
    else:
        return Compose([
            LoadPickle(keys=['tumor'], add_seg=C.seg),
            SpatialPadd(keys=['tumor'], spatial_size=C.sz_t),
            NormalizeIntensityd(keys=['tumor'], nonzero=False, channel_wise=True),
            EnsureTyped(keys=['tumor', 'gt'], dtype=torch.float),
        ])
C.trans_train = _t('train')
C.trans_test = _t('test')

C.y_encode = lambda d: [0,1] if d['转移的前哨（蓝染）LN数量'] > 0 else [1,0]
