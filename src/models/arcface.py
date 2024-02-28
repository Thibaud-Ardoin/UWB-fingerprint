import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

class ArcFace(nn.Module):
    """ Implement of ArcFace: https://arxiv.org/abs/1801.07698 :
        Args:
            input: size of each input
            output: size of each output
            s: scale factor
            m: margin
    """

    def __init__(self, input, output, s=64, m=0.5):
        super(ArcFace, self).__init__()
        self.input = input
        self.output = output
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(output, input))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        cosine = self.calculate_cos(input)
        phi = self.calculate_phi(cosine)
        output = self.calculate_out(cosine, phi, label)
        return output * self.s

    def calculate_cos(self, input):
        return F.linear(F.normalize(input), F.normalize(self.weight))

    def calculate_phi(self, cosine):
        sine = torch.sqrt(torch.clamp((1.0 - torch.pow(cosine, 2)), 1e-9, 1)) # clamp to avoid nan when backward
        return cosine * self.cos_m - sine * self.sin_m

    def calculate_out(self, cosine, phi, label):
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        if label is not None:
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            output = torch.where(one_hot == 1, phi, cosine)
        else:
            output = cosine
        return output