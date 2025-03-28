import torch
import torch.nn as nn


class HybridLoss(nn.Module):
    def __init__(self, weight):
        super(HybridLoss, self).__init__()
        self.spatial = torch.nn.L1Loss()
        self.spectral = SAMLoss(weight=weight)
        self.gra = torch.nn.L1Loss()
        self.weight_gra = weight/10.0

    def forward(self, y, gt):
        spatial_loss = self.spatial(y, gt)
        spectral_loss = self.spectral(y, gt)
        gra_loss = self.weight_gra * self.gra(cal_gradient(y), cal_gradient(gt))
        total_loss = spatial_loss + spectral_loss + gra_loss
        return total_loss


class SAMLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(SAMLoss, self).__init__()
        self.SAMLoss_weight = weight

    def forward(self, x, gt):
        esp = 1e-7
        batch_size = gt.size()[0]
        h_x = gt.size()[2]
        w_x = gt.size()[3]
        Itrue = gt.clone()
        Ifake = x.clone()
        nom = torch.mul(Itrue, Ifake).sum(dim=1)
        denominator = Itrue.norm(p=2, dim=1, keepdim=True).clamp(min=esp) * \
                      Ifake.norm(p=2, dim=1, keepdim=True).clamp(min=esp)
        denominator = denominator.squeeze()
        sam = torch.div(nom, denominator).clamp(-1 + esp, 1 - esp).acos()
        sam[sam != sam] = 0
        sam_sum = torch.sum(sam) / (batch_size * h_x * w_x)
        return self.SAMLoss_weight * sam_sum


def cal_gradient_c(x):
    c_x = x.size(1)
    g = x[:, 1:, 1:, 1:] - x[:, :c_x - 1, 1:, 1:]
    return g


def cal_gradient_x(x):
    c_x = x.size(2)
    g = x[:, 1:, 1:, 1:] - x[:, 1:, :c_x - 1, 1:]
    return g


def cal_gradient_y(x):
    c_x = x.size(3)
    g = x[:, 1:, 1:, 1:] - x[:, 1:, 1:, :c_x - 1]
    return g


def cal_gradient(inp):
    x = cal_gradient_x(inp)
    y = cal_gradient_y(inp)
    c = cal_gradient_c(inp)
    g = torch.sqrt(torch.pow(x, 2) + torch.pow(y, 2) + torch.pow(c, 2) + 1e-6)
    return g
