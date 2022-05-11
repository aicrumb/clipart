from torch.nn import functional as F
from torchvision.transforms import functional as TF
import torch
import torch.nn as nn
import torchvision.transforms as tvt
from torchvision import transforms
import random

class RandResize(torch.nn.Module):
    def __init__(self,size,chance):
        self.size=size
        self.chance=chance
        super().__init__()
    def forward(self,input):
        chance=torch.rand([])
        if chance<self.chance:
            return F.interpolate(input,size=self.size,mode='bilinear')
        else:
            return input
class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1., cutn_whole_portion = 0.0, cutn_bw_portion = 0.2):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.cutn_whole_portion = cutn_whole_portion
        self.cutn_bw_portion = cutn_bw_portion
        custom_augs = [
            tvt.RandomHorizontalFlip(p=0.5),
            tvt.Lambda(lambda x: x + torch.randn_like(x) * 0.02),
            tvt.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            tvt.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            tvt.RandomGrayscale(p=0.5),
        ]
        self.augs = tvt.Compose(custom_augs)
        self.resize = RandResize(224, .5)
    def forward(self, input):
        input = self.resize(input)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        if self.cutn==1:
            cutouts.append(F.adaptive_avg_pool2d(input, self.cut_size))
            return torch.cat(cutouts)
        cut_1 = round(self.cutn*(1-self.cutn_bw_portion))
        cut_2 = self.cutn-cut_1
        gray = transforms.Grayscale(3)
        if cut_1 >0:
            for i in range(cut_1):
                size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
                offsetx = torch.randint(0, sideX - size + 1, ())
                offsety = torch.randint(0, sideY - size + 1, ())
                cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
                if i < int(self.cutn_bw_portion * cut_1):
                    cutout = gray(cutout)
                cutouts.append(self.augs(F.adaptive_avg_pool2d(cutout, self.cut_size)))
        if cut_2 >0:
            for i in range(cut_2):
                cutout = TF.rotate(input, angle=random.uniform(-10.0, 10.0), expand=True, fill=[1,1,1])
                if i < int(self.cutn_bw_portion * cut_2):
                    cutout =gray(cutout)
                cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)