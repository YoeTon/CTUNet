import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


def default_conv(in_channels, out_channels, kernel_size=3, stride=1, bias=True, groups=1, dilation=1):
    """
        padding corresponding kernel 3
    """
    if dilation == 1:
        return nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size // 2), bias=bias, groups=groups)
    elif dilation == 2:
        return nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=2, bias=bias, dilation=dilation, groups=groups)
    elif dilation == 3:
        return nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=3, bias=bias, dilation=dilation, groups=groups)
    elif dilation == 5:
        return nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=5, bias=bias, dilation=dilation, groups=groups)
    elif dilation == 9:
        return nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=9, bias=bias, dilation=dilation, groups=groups)
    else:
        print('unsupported dilation/kernel')
        return


def flow_warp(input, flow, size):
    out_h, out_w = size

    n, c, h, w = input.size()

    norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)

    h_grid = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
    w_gird = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
    grid = torch.cat((w_gird.unsqueeze(2), h_grid.unsqueeze(2)), 2)

    grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
    grid = grid + flow.permute(0, 2, 3, 1) / norm

    output = F.grid_sample(input, grid, align_corners=False)

    return output


class edge(nn.Module):
    def __init__(self, n_feats):
        super(edge, self).__init__()
        self.conv = nn.Conv2d(n_feats * 2, 2, kernel_size=(3, 3), stride=1, padding=(1, 1))

    def forward(self, out):
        size = out.size()[2:]
        flow = F.interpolate(out, (int(size[0] / 4.), int(size[1] / 4.)), mode='bilinear', align_corners=False)
        flow = F.interpolate(flow, size, mode='bilinear', align_corners=False)
        flow = self.conv(torch.cat([out, flow], 1))
        flow = flow_warp(out, flow, size)

        return out - flow


class ResBlock(nn.Module):
    def __init__(self, n_feats, conv=default_conv):
        super(ResBlock, self).__init__()
        self.conv1 = conv(n_feats, n_feats, kernel_size=3, dilation=2, groups=n_feats)
        self.relu = nn.ReLU(True)
        self.conv2 = conv(n_feats, n_feats, kernel_size=3, dilation=2, groups=n_feats)

    def forward(self, x):
        x_input = x
        x = self.conv2(self.relu(self.conv1(x)))
        out = x_input + x
        return out


class BasicBlock(nn.Sequential):
    def __init__(
            self, conv, in_channels, out_channels, kernel_size, stride=1, bias=True, group=1, dilation=1,
            bn=False, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias, dilation=dilation)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


class MSSC(nn.Module):
    def __init__(self, n_feats, bias=None):
        super(MSSC, self).__init__()
        self.input_conv = nn.Conv2d(n_feats, 3 * n_feats, kernel_size=1)

        self.upscale = nn.ConvTranspose2d(n_feats, n_feats, kernel_size=2, stride=2, padding=0, bias=bias)
        self.up_conv1 = nn.Conv2d(n_feats, n_feats, kernel_size=1)
        self.up_conv2 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=2, dilation=2, groups=n_feats)
        self.upback = nn.Conv2d(n_feats, n_feats, kernel_size=2, stride=2, bias=bias)

        self.down_conv1 = nn.Conv2d(n_feats, n_feats, kernel_size=1)
        self.down_conv2 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, groups=n_feats)

        self.mainscale_conv1 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, bias=bias)
        self.relu = nn.ReLU(inplace=False)
        self.mainscale_conv2 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, bias=bias)

        self.omega = nn.Parameter(torch.ones(3))
        self.bn = nn.BatchNorm2d(n_feats)

        self.fuse = nn.Conv2d(3 * n_feats, n_feats, kernel_size=1, bias=bias)

    def forward(self, x):
        x_input = x

        x = self.input_conv(x)
        x_up, x, x_down = x.chunk(3, dim=1)

        x_up = self.upscale(x_up)
        x_up = torch.sigmoid(self.up_conv1(x_up)) * self.up_conv2(x_up)
        x_up = self.upback(x_up)

        x_down = torch.sigmoid(self.down_conv1(x_down)) * self.down_conv2(x_down)

        x_main = self.mainscale_conv2(self.relu(self.bn(self.mainscale_conv1(x))))

        x = torch.cat((self.omega[0] * x_up, self.omega[1] * x_main, self.omega[2] * x_down), dim=1)
        x = self.fuse(x)

        out = x + x_input

        return out


class EGSABlock(nn.Module):
    def __init__(self, n_feats, num_heads=4):
        super(EGSABlock, self).__init__()
        self.spenl = EGSA(dim=n_feats, num_heads=num_heads, bias=False)

    def forward(self, x):
        x_input = x
        x_spe = self.spenl(x)
        x = x_spe
        out = x_input + x
        return out


class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pointwise = nn.Conv2d(dim, dim, 1)
        self.depthwise = nn.Conv2d(dim, dim, 3, padding=2,dilation=2, groups=dim)
        self.depthwise_dilated = nn.Conv2d(dim, dim, 5, stride=1, padding=6, groups=dim, dilation=3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        u = x.clone()
        attn = self.pointwise(x)
        attn = self.depthwise(attn)
        attn = self.depthwise_dilated(attn)
        # attn = self.sig(attn)
        attn = self.avg_pool(attn)
        attn = self.conv(attn.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        attn = 1 + self.sigmoid(attn)
        attn = attn.expand_as(x)
        return u * attn


class VAB(nn.Module):
    def __init__(self, n_feats):
        super(VAB, self).__init__()
        self.proj_1 = nn.Conv2d(n_feats, n_feats, kernel_size=1)
        self.activation = nn.GELU()
        self.atten_branch = Attention(n_feats)
        self.proj_2 = nn.Conv2d(n_feats, n_feats, kernel_size=1)

    def forward(self, x):
        shorcut = x.clone()

        x = self.proj_1(x)
        x = self.activation(x)
        x = self.atten_branch(x)
        x = self.proj_2(x)
        x = x + shorcut

        return x


class EGTB(nn.Module):
    def __init__(self, n_feats, num_heads=4):
        super().__init__()
        self.dim = n_feats
        self.num_heads = num_heads

        self.attn = EGSABlock(n_feats=n_feats, num_heads=num_heads)
        self.drop_path = DropPath(0.1)

        self.ffn = VAB(n_feats)

    def forward(self, x):
        x_input = x

        x = self.attn(x)
        x = self.drop_path(x)
        x = self.ffn(x)

        out = x
        return out


class CTBlock(nn.Module):
    def __init__(self, n_feats, num_heads=4):
        super(CTBlock, self).__init__()
        self.MultiConv = MSSC(n_feats, bias=None)
        self.drop = nn.Dropout(0.1)
        self.TransBlock = EGTB(n_feats=n_feats, num_heads=num_heads)

    def forward(self, x):
        x_input = x

        x = self.MultiConv(x)
        x = self.drop(x)
        x = self.TransBlock(x)

        out = x + x_input
        return out


class EGSA(nn.Module):
    """global spectral attention (DMSA)
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads
        bias (bool): If True, add a learnable bias to projection
    """

    def __init__(self, dim, num_heads, bias):
        super(EGSA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.conv0 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.edge_body = edge(dim)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        b, c, h, w = x.shape
        x_edge = self.edge_body(x)
        x_q = x + 0.01 * x_edge
        q = self.conv0(x_q)
        k = q
        v = self.conv1(x)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, method='deconv', bn=False, act=None, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                if method == 'deconv':
                    m.append(nn.ConvTranspose2d(n_feats, n_feats, kernel_size=2, stride=2, padding=0, bias=bias))
                elif method == 'espcn':
                    m.append(conv(n_feats, 4 * n_feats, 3, bias))
                    m.append(nn.PixelShuffle(2))
                elif method == 'idw':
                    m.append(nn.Upsample(scale_factor=2))
                    m.append(conv(n_feats, n_feats, 3, bias))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act is not None:
                    m.append(act)

        elif scale == 3:
            if method == 'deconv':
                m.append(nn.ConvTranspose2d(n_feats, n_feats, kernel_size=3, stride=3, padding=0, bias=bias))
            elif method == 'espcn':
                m.append(conv(n_feats, 9 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(3))
            elif method == 'idw':
                m.append(nn.Upsample(scale_factor=3))
                m.append(conv(n_feats, n_feats, 3, bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act is not None:
                m.append(act)
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

# if __name__ == '__main__':
#     b = ESSG(block=ESSB, conv=default_conv, n_feats=64, kernel_size=3, n_blocks=2, dilations=[1,2],
#          expand_ratio=2, bias=False, act=None, res=False, attn=ESALayer(k_size=5))
#     print(b)
