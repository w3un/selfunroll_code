
import torch
import torch.nn as nn
from torch.nn import functional as F
from Networks.warp import bwarp
from Networks.base_block import Unet




def pixel_reshuffle(input, upscale_factor):
    r"""Rearranges elements in a tensor of shape ``[*, C, H, W]`` to a
    tensor of shape ``[C*r^2, H/r, W/r]``.

    See :class:`~torch.nn.PixelShuffle` for details.

    Args:
        input (Variable): Input
        upscale_factor (int): factor to increase spatial resolution by

    Examples:
        >>> input = autograd.Variable(torch.Tensor(1, 3, 12, 12))
        >>> output = pixel_reshuffle(input,2)
        >>> print(output.size())
        torch.Size([1, 12, 6, 6])
    """
    batch_size, channels, in_height, in_width = input.size()

    # // division is to keep data type unchanged. In this way, the out_height is still int type
    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor
    input_view = input.contiguous().view(batch_size, channels, out_height, upscale_factor, out_width, upscale_factor)
    channels = channels * upscale_factor * upscale_factor

    shuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return shuffle_out.view(batch_size, channels, out_height, out_width)


class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.LeakyReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c * G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x


class SelfUnroll_plus(nn.Module):
    def __init__(self, D, inchannels):
        super(SelfUnroll_plus, self).__init__()
        self.G0 = 96
        kSize = 3
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.leakyrelu = nn.LeakyReLU()
        self.ELU = nn.ELU()
        self.tanh = nn.Tanh()
        # number of RDB blocks, conv layers, out channels
        self.inchannels = inchannels
        self.D = D
        self.C = 5
        self.G = 48

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d((3 * self.inchannels + 3) * 4, self.G0, 5, padding=2, stride=1)
        self.SFENet2 = nn.Conv2d(self.G0, self.G0, kSize, padding=(kSize - 1) // 2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0=self.G0, growRate=self.G, nConvLayers=self.C)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * self.G0, self.G0, 1, padding=0, stride=1),
            nn.Conv2d(self.G0, self.G0, kSize, padding=(kSize - 1) // 2, stride=1)
        ])

        # Up-sampling net
        self.UPNet = nn.Sequential(*[
            nn.Conv2d(self.G0, 256, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.PixelShuffle(2),
            nn.Conv2d(64, 3, kSize, padding=(kSize - 1) // 2, stride=1)
        ])
        # self.act = nn.Softplus()

        self.flownet = Unet(in_ch=2 * inchannels + 3, out_ch=2, base_chs=32,
                            depth=4)
        self.fusion_net = Unet(in_ch=self.inchannels + 3 * 3 + 2, out_ch=3, base_chs=32,
                               depth=3)
        # self.imgfusion_net = Unet(in_ch=2*self.inchannels + 2 * 3 + 2 * 2, out_ch=3, base_chs=32,
        #                        depth=3)
        self.attfusion_net = Unet(in_ch=2*self.inchannels + 2 * 3 + 2 * 2, out_ch=1, base_chs=32,
                               depth=3)
        self.hardsigmod = nn.Hardsigmoid()


    def forward(self, event1, event2, event3, RS):

        B_shuffle = pixel_reshuffle(torch.cat((RS, event1, event2, event3), 1), 2)
        f__1 = self.SFENet1(B_shuffle)
        x = self.SFENet2(f__1)
        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)
        x1 = self.GFF(torch.cat(RDBs_out, 1))
        x1 = x1 + f__1
        E = self.UPNet(x1)
        E = self.tanh(E)
        sysout = RS + E

        rs2gs_flow = self.flownet(torch.cat((event1, event2, RS), 1))
        rs2gs_flow = self.tanh(rs2gs_flow) * 100.
        flowout = bwarp(RS, rs2gs_flow)

        fusion_out = self.fusion_net(torch.cat((sysout, flowout, rs2gs_flow, event3, RS), 1))
        fusion_out1,fusion_out2 = torch.chunk(fusion_out,2,0)
        event11,event12=torch.chunk(event1,2,0)
        rs2gs_flow1,rs2gs_flow2=torch.chunk(rs2gs_flow,2,0)
        E1,E2 = torch.chunk(E,2,0)
        ###########
        attention_scores = self.attfusion_net(torch.cat((E1,E2,event11,event12,rs2gs_flow1,rs2gs_flow2),1))
        attention = self.hardsigmod(attention_scores)
        out = attention*fusion_out1+(1-attention)*fusion_out2
        return sysout, attention, flowout, rs2gs_flow, fusion_out,out



if __name__ == "__main__":
    batch = 16
    ev1 = torch.rand((batch, 32, 128, 128)).float().cuda()
    ev2 = torch.rand((batch, 32, 128, 128)).float().cuda()
    ev3 = torch.rand((batch, 32, 128, 128)).float().cuda()
    img = torch.rand((batch, 3, 128, 128)).float().cuda()
    unroll_net = SelfUnroll_plus(8, 32)
    unroll_net = unroll_net.cuda()
    sysout, E, flowout, flow, out = unroll_net(ev1, ev2, ev3, img)
    out = sysout
