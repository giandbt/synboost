# CODE FROM NVIDIA Segmentation repositories
import torch.nn as nn
import torch.nn.functional as F

## Creates SPADE normalization layer based on the given configuration
## SPADE consists of two steps. First, it normalizes the activations using
## your favorite normalization method, such as Batch Norm or Instance Norm.
## Second, it applies scale and bias to the normalized output, conditioned on
## the segmentation map.
## |norm_nc|: the #channels of the normalized activations, hence the output dim of SPADE
## |label_nc|: the #channels of the input semantic map, hence the input dim of SPADE
class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()
        
        ks = 3
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        
        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128
        
        pw = ks // 2
        self.shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU())
        self.gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
    
    def forward(self, x, segmap):
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)
        
        # Part 2. produce scaling and bias conditioned on semantic map
        if x.size()[2] != segmap.size()[2]:
            segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.shared(segmap)
        gamma = self.gamma(actv)
        beta = self.beta(actv)
        
        # apply scale and bias
        out = normalized * (1 + gamma) + beta
        
        return out


class FILM(nn.Module):
    def __init__(self, nc, guide_nc):
        super().__init__()
        
        ks = 3
        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128
        
        pw = ks // 2
        self.shared = nn.Sequential(
            nn.Conv2d(guide_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU())
        self.gamma = nn.Conv2d(nhidden, nc, kernel_size=ks, padding=pw)
        self.beta = nn.Conv2d(nhidden, nc, kernel_size=ks, padding=pw)
    
    def forward(self, x, guide):
        if x.size()[2] != guide.size()[2]:
            guide = F.interpolate(guide, size=x.size()[2:], mode='nearest')
        actv = self.shared(guide)
        gamma = self.gamma(actv)
        beta = self.beta(actv)
        
        # apply scale and bias
        out = affine_transformation(x, gamma, beta)
        
        return out


class GuideCorrelation(nn.Module):
    def __init__(self, nc, guide_nc):
        super().__init__()
        
        ks = 3
        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128
        
        pw = ks // 2
        self.shared = nn.Sequential(
            nn.Conv2d(guide_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU())
        self.gamma = nn.Conv2d(nhidden, nc, kernel_size=ks, padding=pw)
        self.beta = nn.Conv2d(nhidden, nc, kernel_size=ks, padding=pw)
    
    def forward(self, x, guide):
    
        # Part 2. produce scaling and bias conditioned on semantic map
        if x.size()[2] != guide.size()[2]:
            guide = F.interpolate(guide, size=x.size()[2:], mode='nearest')
        actv = self.shared(guide)
        gamma = self.gamma(actv)
        beta = self.beta(actv)
        
        return gamma, beta

class GuideNormalization(nn.Module):
    def __init__(self, nc):
        super().__init__()
        
        self.param_free_norm = nn.InstanceNorm2d(nc, affine=False)
    
    def forward(self, x, gamma1, beta1, gamma2, beta2):
        normalized = self.param_free_norm(x)

        gamma = gamma1 * gamma2
        beta = beta1*beta2
        out = normalized * (1 + gamma) + beta
        
        return out


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def affine_transformation(X, alpha, beta):
    x = X.clone()
    mean, std = calc_mean_std(x)
    mean = mean.expand_as(x)
    std = std.expand_as(x)
    return alpha * ((x - mean) / std) + beta