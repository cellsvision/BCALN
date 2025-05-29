import torch
import torch.nn as nn
from monai.networks.nets import DenseNet121


import torch
import torch.nn as nn
import torch.nn.functional as F

class XNet(nn.Module): 
    def __init__(self,cfg):
        super(XNet, self).__init__()

        in_channels_classification = cfg.fea_size

        self.classification_layers = nn.Sequential(
            nn.Linear(in_channels_classification, 512),
            nn.ReLU6(),
            nn.Linear(512, 512),
            nn.ReLU6(),
            nn.Linear(512, 2)
        )

    def forward(self, feat):
        out = self.classification_layers(feat)
        return out



def fetch(cfg, ckpt=None, strict=False, dp=True):
    M = XNet(cfg).to(torch.device("cuda"))
    if ckpt:
        s = torch.load(ckpt, map_location=torch.device("cuda"))
        s = {k.replace('module.', ''): v for k, v in s.items()}
        M.load_state_dict(s, strict=strict)
    return nn.DataParallel(M) if dp else M


