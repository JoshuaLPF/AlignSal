import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from backbone.dfformer import cdfformer_s18, StarReLU, resize_complex_weight

from timm.models.layers.helpers import to_2tuple
from thop import profile
from thop import clever_format


def DWconv3(in_dim, out_dim, stride=1, has_bias=False):
    "3x3 depth-wise convolution with padding"
    return nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=stride, padding=1, groups=out_dim, bias=has_bias)

def DWconv3_BN_ReLU(in_dim, out_dim, stride=1):
    return nn.Sequential(
            DWconv3(in_dim, out_dim, stride),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),)



class EfficientDynamicFilter(nn.Module):
    def __init__(self, dim, size, act1_layer=StarReLU, act2_layer=nn.Identity, num_filters=4, weight_resize=False,
                 **kwargs):
        super().__init__()
        size = to_2tuple(size)
        self.size = size[0]
        self.filter_size = size[1] // 2 + 1
        self.num_filters = num_filters
        self.dim = dim
        self.med_channels = int(num_filters * dim)
        self.weight_resize = weight_resize
        self.act1 = act1_layer()
        self.complex_weights = nn.Parameter(torch.randn(self.size, self.filter_size, num_filters, 2, dtype=torch.float32) * 0.02) # gfnet
        self.act2 = act2_layer()

        self.pwconv1 = nn.Conv2d(dim, self.med_channels, kernel_size=1, stride=1, padding=0, groups=dim, bias=False)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.bnorm = nn.BatchNorm2d(dim)
        self.Dwconv_ffn = DWconv3(dim, dim)

    def forward(self, x, y):
        B, _, _, _ = x.shape
        shortcut = y + x
        x_norm = self.bnorm(x)
        y_norm = self.bnorm(y)

        f_norm = x_norm + y_norm
        f_pre = self.pwconv1(f_norm)
        routeing = self.max_pool(f_pre).view(B, self.num_filters, -1).softmax(dim=1)

        f_norm = self.act1(f_norm)
        f_float = f_norm.to(torch.float32)

        f_fft = torch.fft.rfft2(f_float, dim=(2, 3), norm='ortho')

        if self.weight_resize:
            complex_weights = resize_complex_weight(self.complex_weights, f_fft.shape[2],
                                                    f_fft.shape[3])
            complex_weights = torch.view_as_complex(complex_weights.contiguous())
        else:
            complex_weights = torch.view_as_complex(self.complex_weights)
        routeing = routeing.to(torch.complex64)
        weight = torch.einsum('bfc,hwf->bchw', routeing, complex_weights)
        f_fft = f_fft * weight
        f = torch.fft.irfft2(f_fft, dim=(2, 3), norm='ortho')
        f = self.act2(f)
        f = f + shortcut

        f_ffn = self.bnorm(f)
        f_ffn = self.Dwconv_ffn(f_ffn)
        f = f + f_ffn

        return f


class AlignSal(nn.Module):
    def __init__(self):
        super(AlignSal, self).__init__()
        # encoder
        self.encoder = cdfformer_s18()

        self.EDF4 = EfficientDynamicFilter(512, 12)
        self.EDF3 = EfficientDynamicFilter(320, 24)
        self.EDF2 = EfficientDynamicFilter(128, 48)
        self.EDF1 = EfficientDynamicFilter(64, 96)

        # decoder
        self.upsample1 = nn.Sequential(DWconv3_BN_ReLU(124, 31),
                                       nn.UpsamplingBilinear2d(scale_factor=2, ))
        self.upsample2 = nn.Sequential(DWconv3_BN_ReLU(240, 60),
                                       nn.UpsamplingBilinear2d(scale_factor=2, ))
        self.upsample3 = nn.Sequential(DWconv3_BN_ReLU(448, 112),
                                       nn.UpsamplingBilinear2d(scale_factor=2, ))
        self.upsample4 = nn.Sequential(DWconv3_BN_ReLU(512, 128),
                                       nn.UpsamplingBilinear2d(scale_factor=2, ))

        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.S4 = nn.Conv2d(144, 1, 3, stride=1, padding=1)
        self.S3 = nn.Conv2d(132, 1, 3, stride=1, padding=1)
        self.S2 = nn.Conv2d(156, 1, 3, stride=1, padding=1)
        self.S1 = nn.Conv2d(132, 1, 3, stride=1, padding=1)

        self.dwconv512_r = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1, groups=1, bias=False)
        self.dwconv512_t = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1, groups=1, bias=False)

        self.pred = nn.Conv2d(31, 1, kernel_size=3, stride=1, padding=1, groups=1, bias=False)

    def forward(self, rgb, thermal,):
        r = self.encoder(rgb)
        r1 = r[0]
        r2 = r[1]
        r3 = r[2]
        r4 = r[3]

        t = self.encoder(thermal)
        t1 = t[0]
        t2 = t[1]
        t3 = t[2]
        t4 = t[3]

        # For Contrastive Learning
        r4_cl = self.dwconv512_r(r4)
        t4_cl = self.dwconv512_r(t4)

        # Fusion Part
        f1 = self.EDF1(r1, t1)
        f2 = self.EDF2(r2, t2)
        f3 = self.EDF3(r3, t3)
        f4 = self.EDF4(r4, t4)

        # Decoder
        F4 = self.upsample4(f4)
        F3 = torch.cat((f3, F4), dim=1)
        F3 = self.upsample3(F3)
        F2 = torch.cat((f2, F3), dim=1)
        F2 = self.upsample2(F2)
        F1 = torch.cat((f1, F2), dim=1)
        F1 = self.upsample1(F1)

        # saliency prediction
        out = self.pred(F1)
        return out, r4_cl, t4_cl

    def load_pre(self, pre_model):
        self.rgb_feature.load_state_dict(torch.load(pre_model), strict=False)
        print(f"RGB loading pre_model ${pre_model}")
        self.depth_feature.load_state_dict(torch.load(pre_model), strict=False)
        print(f"Depth loading pre_model ${pre_model}")


if __name__ == '__main__':
    a = torch.randn(1, 3, 384, 384)
    b = torch.randn(1, 3, 384, 384)
    c = torch.Tensor(a).cuda()
    d = torch.Tensor(b).cuda()
    model = AlignSal().cuda()
    e = model(c, d)
    flops, params = profile(model, inputs=(c, d))
    flops, params = clever_format([flops, params], '%.3f')
    print(flops, params)