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
C.f_d = './sample_data/fd'
C.workdir = './ckpts/c'
C.log = 'log_xx'
C.seg = 'seg_crop'
C.bz = 24
C.eps = 300
C.sz_t = [100,100,50]
C.sz_a = [100,140,70]
C.fea_size = 2048
C.lr = 1e-4
C.opt = Adan
C.loss = nn.CrossEntropyLoss()
C.ckpt = './ckpts/t_a_slnn.pth'
C.excl = []

def _t(z):
    if z == 'train':
        return Compose([
            LoadPickle(keys=['tumor'], add_seg=C.seg),
            LoadPickle(keys=['axillary_left', 'axillary_right'], add_seg='no_seg'),
            SpatialPadd(keys=['tumor'], spatial_size=C.sz_t),
            SpatialPadd(keys=['axillary_left', 'axillary_right'], spatial_size=C.sz_a),
            CombineSeq(keys=['axillary_left', 'axillary_right'], final_key_name='axillary', axis=1),
            OneOf([ 
                RandScaleIntensityd(keys=["tumor","axillary"],prob=0.2,factors=0.4), 
                RandAdjustContrastd(keys=["tumor","axillary"],prob=0.2), 
                RandHistogramShiftd(keys=["tumor","axillary"],prob=0.2), 
                RandShiftIntensityd(keys=["tumor","axillary"],prob=0.2,offsets=0.2),
                RandGaussianNoised(keys=["tumor","axillary"], prob=0.2),
                RandGaussianSmoothd(keys=["tumor","axillary"], prob=0.2),
                RandGaussianSharpend(keys=["tumor","axillary"], prob=0.2),
            ]),
            NormalizeIntensityd(keys=['tumor'], nonzero=False, channel_wise=True),
            NormalizeIntensityd(keys=['axillary'], nonzero=False, channel_wise=True),
            EnsureTyped(keys=['tumor', 'axillary', 'gt'], dtype=torch.float),
        ])
    else:
        return Compose([
            LoadPickle(keys=['tumor'], add_seg=C.seg),
            LoadPickle(keys=['axillary_left', 'axillary_right'], add_seg='no_seg'),
            SpatialPadd(keys=['tumor'], spatial_size=C.sz_t),
            SpatialPadd(keys=['axillary_left', 'axillary_right'], spatial_size=C.sz_a),
            CombineSeq(keys=['axillary_left', 'axillary_right'], final_key_name='axillary', axis=1),
            NormalizeIntensityd(keys=['tumor'], nonzero=False, channel_wise=True),
            NormalizeIntensityd(keys=['axillary'], nonzero=False, channel_wise=True),
            EnsureTyped(keys=['tumor', 'axillary', 'gt'], dtype=torch.float),
        ])
C.trans_train = _t('train')
C.trans_test = _t('test')

C.y_encode = lambda x: [0,1] if x['positive_SLN_count']>=3 else [1,0]
