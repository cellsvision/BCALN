import torch
import torch.nn as nn
from monai.networks.nets import DenseNet121


class AttMIL(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Tanh(),
            nn.Linear(out_features, 1)
        )
        self.softmax = nn.Softmax(dim=1)

        self._init_w()

    def _init_w(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        attn = self.attention(x)  
        weights = self.softmax(attn) 
        weighted = (weights * x).sum(dim=1)  
        return weighted


class XNet(nn.Module):
    def __init__(self,a):
        super().__init__()
        f = lambda: DenseNet121(spatial_dims=3, in_channels=3, out_channels=1, dropout_prob=0.3)
        self.encoder_tumor = f()
        self.encoder_axillary = f()
        self.encoder_tumor.class_layers = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten())
        self.encoder_axillary.class_layers = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten())
        self.classification_layers = nn.Sequential(nn.Linear(2048, 512), nn.ReLU6(), nn.Linear(512, 2))
        self.a = a
        if a:
            self.attmil = AttMIL(in_features=1024, out_features=1024)
    def forward(self, x1, x2):
        if self.a:
            return self.classification_layers(torch.cat([self.attmil(self.encoder_tumor(x1)), self.encoder_axillary(x2)], dim=1))
        else:
            return self.classification_layers(torch.cat([self.encoder_tumor(x1), self.encoder_axillary(x2)], dim=1))


class XNet_t(nn.Module):
    def __init__(self):
        super().__init__()
        f = lambda: DenseNet121(spatial_dims=3, in_channels=3, out_channels=1, dropout_prob=0.3)
        self.encoder_tumor = f()
    def forward(self, x1):
        return self.encoder_tumor(x1)


class XNet_a(nn.Module):
    def __init__(self):
        super().__init__()
        f = lambda: DenseNet121(spatial_dims=3, in_channels=3, out_channels=1, dropout_prob=0.3)
        self.encoder_axillary = f()
    def forward(self, x1):
        return self.encoder_axillary(x1)

def fetch(ckpt=None, strict=False, dp=True, mod='ta'):
    if mod=='ta':
        M = XNet(a=False).to(torch.device("cuda"))
    elif mod=='t':
        M = XNet_t().to(torch.device("cuda"))
    elif mod=='a':
        M = XNet_a().to(torch.device("cuda"))
    else:
        raise NotImplementedError
    if ckpt:
        s = torch.load(ckpt, map_location=torch.device("cuda"))
        s = {k.replace('module.', ''): v for k, v in s.items()}
        M.load_state_dict(s, strict=strict)
    return nn.DataParallel(M) if dp else M


