import torch, os
import torch.nn as nn
from . import pretrained_model as pn

# from https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/lpips.py
def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2,3],keepdim=keepdim)

def upsample(in_tens, out_HW=(64,64)): # assumes scale factor is same for H and W
    in_H, in_W = in_tens.shape[2], in_tens.shape[3]
    return nn.Upsample(size=out_HW, mode='bilinear', align_corners=False)(in_tens)

# Learned perceptual metric
class LPIPS(nn.Module):
    def __init__(
        self, net='vgg', pretrained=True, spatial=False, use_dropout=True, eval_mode=True,
        cache_dir=None,
    ):
        super(LPIPS, self).__init__()
        self.pnet_type = net
        self.spatial = spatial
        self.scaling_layer = ScalingLayer()

        if(self.pnet_type in ['vgg','vgg16']):
            net_type = pn.VGG16
            self.chns = [64,128,256,512,512]
        else:
            raise RuntimeError('currently only vgg16 supported!')
        self.L = len(self.chns)

        _dir = torch.hub.get_dir()
        if cache_dir is not None: torch.hub.set_dir(cache_dir)
        self.net = net_type(pretrained=pretrained)
        torch.hub.set_dir(_dir)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.lins = [self.lin0,self.lin1,self.lin2,self.lin3,self.lin4]
        if(self.pnet_type=='squeeze'): # 7 layers for squeezenet
            self.lin5 = NetLinLayer(self.chns[5], use_dropout=use_dropout)
            self.lin6 = NetLinLayer(self.chns[6], use_dropout=use_dropout)
            self.lins+=[self.lin5,self.lin6]
        self.lins = nn.ModuleList(self.lins)
        lins_model_path = os.path.join(os.path.dirname(__file__), 'vgg_v0.1.pth')
        self.load_state_dict(torch.load(lins_model_path, map_location='cpu'), strict=False)

        if(eval_mode):
            self.eval()

    def forward(self, in0, in1, retPerLayer=False, normalize=False):
        if normalize: # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            in0 = 2 * in0  - 1
            in1 = 2 * in1  - 1

        # v0.0 - original release had a bug, where input was not scaled
        in0_input, in1_input = (self.scaling_layer(in0), self.scaling_layer(in1))
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}

        for kk in range(self.L):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk]-feats1[kk])**2

        if(self.spatial):
            res = [upsample(self.lins[kk](diffs[kk]), out_HW=in0.shape[2:]) for kk in range(self.L)]
        else:
            res = [spatial_average(self.lins[kk](diffs[kk]), keepdim=True) for kk in range(self.L)]

        val = 0
        for l in range(self.L):
            val += res[l]

        if(retPerLayer):
            return (val, res)
        else:
            return val

class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030,-.088,-.188])[None,:,None,None])
        self.register_buffer('scale', torch.Tensor([.458,.448,.450])[None,:,None,None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale

class NetLinLayer(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = [nn.Dropout(),] if(use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def normalize_tensor(in_feat,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
    return in_feat/(norm_factor+eps)
