import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import colorsys


# class mse_loss(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.loss_fn = nn.MSELoss()
#     def forward(self, output, target):
#         return self.loss_fn(output, target)


def mse_loss(output, target):
    return F.mse_loss(output, target)

import torch

def rgb_to_hsv(x):
    # Convert color space
    x = torch.clamp(x, 0.0, 1.0) # Clip values to [0, 1]
    x = x.view(-1, 3, x.shape[2], x.shape[3]) # Reshape to [N, 3, H, W]
  
    # Get the max, min and difference of RGB channels
    max_rgb, _ = x.max(dim=1) # Shape: [N, H, W]
    min_rgb, _ = x.min(dim=1) # Shape: [N, H, W]
    diff_rgb = max_rgb - min_rgb # Shape: [N, H, W]

    # Compute the hue channel
    hue = torch.zeros_like(max_rgb) # Shape: [N, H, W]
    mask_r = (max_rgb == x[:, 0]) & (diff_rgb != 0) # Mask for red channel
    mask_g = (max_rgb == x[:, 1]) & (diff_rgb != 0) # Mask for green channel
    mask_b = (max_rgb == x[:, 2]) & (diff_rgb != 0) # Mask for blue channel
    hue[mask_r] = ((x[:, 1] - x[:, 2])[mask_r] / diff_rgb[mask_r]) % 6
    hue[mask_g] = ((x[:, 2] - x[:, 0])[mask_g] / diff_rgb[mask_g]) + 2
    hue[mask_b] = ((x[:, 0] - x[:, 1])[mask_b] / diff_rgb[mask_b]) + 4
    hue = hue / 6 # Normalize to [0, 1]

    # Compute the saturation channel
    sat = torch.zeros_like(max_rgb) # Shape: [N, H, W]
    mask_nonzero = max_rgb != 0 # Mask for nonzero max values
    sat[mask_nonzero] = diff_rgb[mask_nonzero] / max_rgb[mask_nonzero]

    # Compute the value channel
    val = max_rgb

    # Concatenate the HSV channels
    x_hsv = torch.stack([hue, sat, val], dim=1) # Shape: [N, 3, H, W]

    return x_hsv


class ColorHistogramLoss(nn.Module):
    """颜色直方图损失函数"""
    def __init__(self, alpha=0.3, beta=0.4, gamma=0.4):
        super(ColorHistogramLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, x_real, x_fake):
        # 转换颜色空间
        x_real_hsv = rgb_to_hsv(x_real)
        x_fake_hsv = rgb_to_hsv(x_fake)
        # 计算H通道的直方图分布
        real_hist_h = torch.histc(x_real_hsv[:, 0], bins=10, min=0, max=1).cuda() 
        fake_hist_h = torch.histc(x_fake_hsv[:, 0], bins=10, min=0, max=1).cuda()
        # 计算S通道的直方图分布
        real_hist_s = torch.histc(x_real_hsv[:, 1], bins=10, min=0, max=1).cuda() 
        fake_hist_s = torch.histc(x_fake_hsv[:, 1], bins=10, min=0, max=1).cuda()
        # 计算V通道的直方图分布
        real_hist_v = torch.histc(x_real_hsv[:, 2], bins=10, min=0, max=1).cuda() 
        fake_hist_v = torch.histc(x_fake_hsv[:, 2], bins=10, min=0, max=1).cuda()
        # 计算各个通道的直方图差异
        hist_loss_h = torch.mean(torch.abs(real_hist_h - fake_hist_h)).cuda() 
        hist_loss_s = torch.mean(torch.abs(real_hist_s - fake_hist_s)).cuda() 
        hist_loss_v = torch.mean(torch.abs(real_hist_v - fake_hist_v)).cuda()
        # 计算最终的颜色直方图损失函数
        hist_loss = hist_loss_h * self.alpha + hist_loss_s * self.beta + hist_loss_v * self.gamma 
        return hist_loss



class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss



class Gradient_Loss(nn.Module):
    def __init__(self):
        super(Gradient_Loss, self).__init__()

        kernel_g = [[[0,1,0],[1,-4,1],[0,1,0]],
                    [[0,1,0],[1,-4,1],[0,1,0]],
                    [[0,1,0],[1,-4,1],[0,1,0]]]
        kernel_g = torch.FloatTensor(kernel_g).unsqueeze(0).permute(1, 0, 2, 3)
        self.weight_g = nn.Parameter(data=kernel_g, requires_grad=False)

    def forward(self, x,xx):
        grad = 0
        y = x
        yy = xx
        gradient_x = F.conv2d(y, self.weight_g,groups=3)
        gradient_xx = F.conv2d(yy,self.weight_g,groups=3)
        l = nn.L1Loss()
        a = l(gradient_x,gradient_xx)
        grad = grad + a
        return grad


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

