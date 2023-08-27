import torch
import torch.nn as nn
import math
from utils import  get_mask_from_data
import torch.nn.functional as F
from qsm_fun import get_tkd


class AffineResBlock(nn.Module):
    ''' mod from AffineResBlock '''
    def __init__(self, in_ch, out_ch, stride1=1, stride2=1, k_size=3, ud_flag=None, bias=True):
        super(AffineResBlock, self).__init__()
        self.Conv1 = nn.Conv3d(in_ch, out_ch, k_size, stride=stride1, padding=int(k_size // 2),
                                 padding_mode='replicate', bias=bias)
        self.Conv2 = nn.Conv3d(out_ch, out_ch, k_size, stride=stride2, padding=int(k_size // 2),
                                 padding_mode='replicate', bias=bias)
        self.ud_flag = ud_flag

    def forward(self, inputs):
        if self.ud_flag == 'up':
            x = F.interpolate(inputs, scale_factor=2, mode='trilinear', align_corners=True)
        elif self.ud_flag == 'down':
            x = F.interpolate(inputs, scale_factor=0.5, mode='trilinear', align_corners=True)
        else:
            x = inputs
        x1 = F.leaky_relu(self.Conv1(x))
        x2 = F.leaky_relu(self.Conv2(x1))
        return x1+x2


class FeatureExtraction(nn.Module):
    def __init__(self, in_ch, out_ch=128):
        super(FeatureExtraction, self).__init__()
        self.block1 = AffineResBlock(in_ch, 64, 1, 1, ud_flag='down')
        self.block2 = AffineResBlock(64, out_ch, 1, 1, ud_flag='down')

    def forward(self, inputs):
        x1 = self.block1(inputs)
        x2 = self.block2(x1)
        return x2


class ModulatedConv3d(nn.Module):
    """
        ModulatedConv3d
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_style_feat, demodulate=True, eps=1e-8):
        super(ModulatedConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.eps = eps
        self.linear = nn.Linear(num_style_feat, in_channels)
        self.weight = nn.Parameter(torch.randn(1, out_channels, in_channels, kernel_size, kernel_size, kernel_size) /
                                   math.sqrt(in_channels * kernel_size ** 3))
        self.padding = kernel_size // 2
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1, 1))
        self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, style):
        b, c, h, w, d = x.shape
        style = self.linear(style)
        style = style.view(b, 1, self.in_channels, 1, 1, 1)
        weight = self.weight * style
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4, 5]) + self.eps)
            weight = weight * demod.view(b, self.out_channels, 1, 1, 1, 1)
        weight = weight.view(b * self.out_channels, c, self.kernel_size, self.kernel_size, self.kernel_size)
        b, c, h, w, d = x.shape
        x = x.view(1, b * c, h, w, d)
        out = F.conv3d(x, weight, padding=self.padding, groups=b)
        out = out.view(b, self.out_channels, *out.shape[2:5])
        out = out + self.bias
        out = self.activate(out)
        return out


class Fea2LatentCode(nn.Module):
    """ Feature to Latent Code  """
    def __init__(self, in_ch, out_ch, bias=True):
        super(Fea2LatentCode, self).__init__()
        self.c1 = nn.Conv3d(in_ch, min(in_ch, out_ch), (3, 3, 3), (2, 2, 1), (1, 1, 1), bias=bias)
        self.c2 = nn.Conv3d(min(in_ch, out_ch), out_ch, 3, 1, 0, bias=bias)

    def forward(self, inputs, mask):
        x = F.leaky_relu(self.c1(inputs), negative_slope=0.2)
        x1 = F.leaky_relu(self.c2(x), negative_slope=0.2)
        x = torch.sum(torch.abs(x1), dim=[2, 3, 4])/(torch.sum(mask, dim=[2,3,4])+1e-8)
        return x


class FuseBlock(nn.Module):
    def __init__(self,in_ch=32,out_ch=32):
        super(FuseBlock,self).__init__()
        self.c1=nn.Conv3d(in_ch,out_ch,3,1,1,padding_mode='replicate')
        self.c2=nn.Conv3d(in_ch,out_ch,3,1,1,padding_mode='replicate')
        self.c3=nn.Conv3d(in_ch,out_ch,3,1,1,padding_mode='replicate')
        self.c_fuse = nn.Conv3d(3*out_ch,out_ch,1, 1, 0)
        self.c1_out = nn.Conv3d(out_ch, out_ch, 3, 1, 1, padding_mode='replicate')
        self.c2_out = nn.Conv3d(out_ch, out_ch, 3, 1, 1, padding_mode='replicate')
        self.c3_out = nn.Conv3d(out_ch, out_ch, 3, 1, 1, padding_mode='replicate')

    def forward(self, x1, x2, x3):
        x_c1 = self.c1(x1)
        x_c2 = self.c2(x2)
        x_c3 = self.c3(x3)
        x_cat = torch.cat([x_c1, x_c2, x_c3], 1)
        x_fuse = self.c_fuse(x_cat)
        x_out1 = F.leaky_relu(self.c1_out(x_fuse), negative_slope=0.2)
        x_out2 = F.leaky_relu(self.c2_out(x_fuse), negative_slope=0.2)
        x_out3 = F.leaky_relu(self.c3_out(x_fuse), negative_slope=0.2)
        return x_out1, x_out2, x_out3


class MagCombineFuseBlock(nn.Module):
    def __init__(self, in_ch=32, out_ch=32):
        super(MagCombineFuseBlock, self).__init__()
        self.c1 = nn.Conv3d(in_ch, out_ch, 3, 1, 1, padding_mode='replicate')
        self.c2 = nn.Conv3d(in_ch, out_ch, 3, 1, 1, padding_mode='replicate')
        self.c3 = nn.Conv3d(in_ch, out_ch, 3, 1, 1, padding_mode='replicate')
        self.c_fuse = nn.Conv3d(3*out_ch, out_ch, 1, 1, 0)
        self.c_mag1 = nn.Conv3d(1, 16, 3, 1, 1, padding_mode='replicate')
        self.c_mag2 = nn.Conv3d(16, out_ch, 3, 1, 1, padding_mode='replicate')
        self.c1_out = nn.Conv3d(out_ch, out_ch, 3, 1, 1, padding_mode='replicate')
        self.c2_out = nn.Conv3d(out_ch, out_ch, 3, 1, 1, padding_mode='replicate')
        self.c3_out = nn.Conv3d(out_ch, out_ch, 3, 1, 1, padding_mode='replicate')

    def forward(self, x1, x2, x3, mag):
        x_c1 = self.c1(x1)
        x_c2 = self.c2(x2)
        x_c3 = self.c3(x3)
        mask = get_mask_from_data(mag)
        x_cat = torch.cat([x_c1, x_c2, x_c3], 1)
        x_fuse = self.c_fuse(x_cat)
        x_mag = F.leaky_relu(self.c_mag2(F.leaky_relu(self.c_mag1(mag), negative_slope=0.2)), negative_slope=0.2)
        x_fuse_mag = x_mag*mask + x_fuse
        x_out1 = F.leaky_relu(self.c1_out(x_fuse_mag), negative_slope=0.2)
        x_out2 = F.leaky_relu(self.c2_out(x_fuse_mag), negative_slope=0.2)
        x_out3 = F.leaky_relu(self.c3_out(x_fuse_mag), negative_slope=0.2)
        return x_out1, x_out2, x_out3
        # return x_out1, x_out2, x_out3, x_mag, x_fuse_mag, x_fuse, x_c1, x_c2, x_c3


class MagCombineFuseBlock_n(nn.Module):
    '''  MagCombineFuseBlock  '''
    def __init__(self, in_ch=32, out_ch=32, n=3):
        super(MagCombineFuseBlock_n, self).__init__()
        self.branch_in_list = nn.ModuleList()
        self.branch_out_list = nn.ModuleList()
        for i in range(n):
            self.branch_in_list.append(nn.Conv3d(in_ch, out_ch, 3, 1, 1, padding_mode='replicate'))
        self.c_fuse = nn.Conv3d(n*out_ch, out_ch, 1, 1, 0)
        self.c_mag1 = nn.Conv3d(1, 16, 3, 1, 1, padding_mode='replicate')
        self.c_mag2 = nn.Conv3d(16, out_ch, 3, 1, 1, padding_mode='replicate')
        for i in range(n):
            self.branch_out_list.append(nn.Conv3d(in_ch, out_ch, 3, 1, 1, padding_mode='replicate'))

    def forward(self, input_list, mag):
        for i in range(len(input_list)):
            temp = self.branch_in_list[i](input_list[i])
            x_cat = temp if i == 0 else torch.cat([x_cat, temp], dim=1)
        x_fuse = self.c_fuse(x_cat)
        mask = get_mask_from_data(mag)
        x_mag = F.leaky_relu(self.c_mag2(F.leaky_relu(self.c_mag1(mag), negative_slope=0.2)), negative_slope=0.2)
        x_fuse_mag = x_mag*mask + x_fuse
        output_list = []
        for i in range(len(input_list)):
            # in some model file ,the leakey relu slope is default
            output_list.append(F.leaky_relu(self.branch_out_list[i](x_fuse_mag), negative_slope=0.2))
        return output_list


class CrossFusionBlock(nn.Module):
    def __init__(self, in_ch=32, out_ch=32):
        super(CrossFusionBlock, self).__init__()
        self.c1 = nn.Conv3d(in_ch, out_ch, 3, 1, 1, padding_mode='replicate')
        self.c2 = nn.Conv3d(in_ch, out_ch, 3, 1, 1, padding_mode='replicate')
        self.c3 = nn.Conv3d(in_ch, out_ch, 3, 1, 1, padding_mode='replicate')
        self.c_fuse = nn.Conv3d(3*out_ch, out_ch, 1, 1, 0)
        self.c_mag1 = nn.Conv3d(1, 16, 3, 1, 1, padding_mode='replicate')
        self.c_magf_1 = nn.Conv3d(16, out_ch, 3, 1, 1, padding_mode='replicate')
        self.c_magf_2 = nn.Conv3d(16, out_ch, 3, 1, 1, padding_mode='replicate')
        self.c1_out = nn.Conv3d(out_ch, out_ch, 3, 1, 1, padding_mode='replicate')
        self.c2_out = nn.Conv3d(out_ch, out_ch, 3, 1, 1, padding_mode='replicate')
        self.c3_out = nn.Conv3d(out_ch, out_ch, 3, 1, 1, padding_mode='replicate')

    def forward(self, x1, x2, x3, mag):
        x_c1 = self.c1(x1)
        x_c2 = self.c2(x2)
        x_c3 = self.c3(x3)
        mask = get_mask_from_data(mag)
        x_cat = torch.cat([x_c1, x_c2, x_c3], 1)
        x_fuse = self.c_fuse(x_cat)
        x_mag_1 = F.leaky_relu(self.c_mag1(mag), negative_slope=0.2)*mask
        x_magf1 = F.leaky_relu(self.c_magf_1(x_mag_1), negative_slope = 0.2)
        x_magf2 = torch.sigmoid(self.c_magf_2(x_mag_1))  # get the fea mask
        x_fuse_mag = x_fuse*x_magf1*x_magf2+x_fuse*(1-x_magf1)+x_magf1

        x_out1 = F.leaky_relu(self.c1_out(x_fuse_mag), negative_slope=0.2)
        x_out2 = F.leaky_relu(self.c2_out(x_fuse_mag), negative_slope=0.2)
        x_out3 = F.leaky_relu(self.c3_out(x_fuse_mag), negative_slope=0.2)

        return x_out1, x_out2, x_out3
        # return x_out1, x_out2, x_out3, x_fuse,x_fuse_mag


class LCMnet(nn.Module):
    '''
       LCMnet final version
    '''
    def __init__(self, num_style_feat=256, body_ch=32):
        super(LCMnet, self).__init__()
        self.get_deep_feature = FeatureExtraction(body_ch)
        self.tkd_conv = nn.Conv3d(1, body_ch, 3, 1, 1)

        self.mod_conv1 = ModulatedConv3d(body_ch, body_ch, 3, num_style_feat)
        self.a1 = Fea2LatentCode(128, num_style_feat)
        self.b1 = AffineResBlock(body_ch, body_ch, 1, 1)
        
        self.mod_conv2 = ModulatedConv3d(body_ch, body_ch, 3, num_style_feat)
        self.a2 = Fea2LatentCode(128, num_style_feat)
        self.b2 = AffineResBlock(body_ch, body_ch, 1, 1)

        self.mod_conv3 = ModulatedConv3d(body_ch, body_ch, 3, num_style_feat)
        self.a3 = Fea2LatentCode(128, num_style_feat)
        self.b3 = AffineResBlock(1, body_ch)

        self.fuse_block = MagCombineFuseBlock(32, 32)
        self.conv = nn.Conv3d(body_ch, 1, 1, 1, 0, bias=False)

    def forward(self, deltab, mag, params=None, flag=1, s=(224, 224, 126)):
        assert mag.shape == deltab.shape,\
            f"the shape of the mag should be same as the deltab,now deltab.shape == {deltab.shape},mag.shape == {mag.shape}"
        if deltab.shape[4] >= 64:
            ''' test '''
            flag = 0
        tkd = get_tkd(deltab, 0.15, params, flag, s=s)
        b3 = self.b3(deltab)
        b2 = self.b2(b3)
        b1 = self.b1(b2)
        b0 = self.get_deep_feature(b1)
        mask_init = get_mask_from_data(deltab)
        mask = F.interpolate(mask_init, scale_factor=0.25, mode='trilinear', align_corners=False)
        b3_out, b2_out, b1_out = self.fuse_block(b3, b2, b1, mag)
        tkd_conv = F.leaky_relu(self.tkd_conv(tkd))
        a1 = self.a1(b0, mask)
        x1 = self.mod_conv1(tkd_conv, a1) + b1_out

        a2 = self.a2(b0, mask)
        x2 = self.mod_conv2(x1, a2) + b2_out

        a3 = self.a3(b0, mask)
        x3 = self.mod_conv3(x2, a3) + b3_out
        x = self.conv(x3)
        return x


class LCMnet_n(nn.Module):
    '''
       LCMnet with n blocks
    '''
    def __init__(self, num_style_feat=256, body_ch=32, n=5):
        super(LCMnet_n, self).__init__()
        self.get_deep_feature = FeatureExtraction(body_ch)
        self.tkd_conv = nn.Conv3d(1, body_ch, 3, 1, 1)
        self.modulate_conv = nn.ModuleList()
        self.encoder_unit = nn.ModuleList()
        self.decoder_unit = nn.ModuleList()
        for i in range(n):
            self.modulate_conv.append(ModulatedConv3d(body_ch, body_ch, 3, num_style_feat))
            if i == n-1:
                self.encoder_unit.append(AffineResBlock(1, body_ch, 1, 1))
            else:
                self.encoder_unit.append(AffineResBlock(body_ch, body_ch, 1, 1))
            self.decoder_unit.append(Fea2LatentCode(128, num_style_feat))
        self.fuse_block = MagCombineFuseBlock_n(32, 32, n=n)
        self.conv = nn.Conv3d(body_ch, 1, 1, 1, 0, bias=False)

    def forward(self, deltab, mag, params=None, flag=1, s=(224, 224, 126)):
        assert mag.shape == deltab.shape,\
            f"The shape of the mag should be same as the deltab,now deltab.shape == {deltab.shape},mag.shape == {mag.shape}"
        block_number = len(self.modulate_conv)
        if flag == 1:
            if deltab.shape[4] >= 64:
                flag=0
            input_list = []
            temp = deltab
            for i in range(block_number):
                # x3,x2,x1
                temp = self.encoder_unit[-i-1](temp)
                input_list.append(temp)
            b0 = self.get_deep_feature(temp)
            mask_init = get_mask_from_data(deltab)
            mask = F.interpolate(mask_init, scale_factor=0.25, mode='trilinear', align_corners=False)
            output_list = self.fuse_block(input_list, mag)
            tkd = get_tkd(deltab, 0.15, params, flag, s=s)
            tkd_conv = F.leaky_relu(self.tkd_conv(tkd))
            for i in range((block_number)):
                a_temp = self.decoder_unit[i](b0, mask)
                if i == 0:
                    x_temp = self.modulate_conv[i](tkd_conv, a_temp) + output_list[i]
                else:
                    x_temp = self.modulate_conv[i](x_temp, a_temp) + output_list[i]
            x = self.conv(x_temp)
            return x

        else:
            input_list = []
            temp = deltab
            for i in range(block_number):
                # x3,x2,x1
                temp = self.encoder_unit[-i - 1](temp)
                input_list.append(temp)
            b0 = self.get_deep_feature(temp)
            mask_init = get_mask_from_data(deltab)
            mask = F.interpolate(mask_init, scale_factor=0.25, mode='trilinear', align_corners=False)
            output_list = self.fuse_block(input_list, mag)
            tkd = get_tkd(deltab, 0.15, params, flag, s=s)
            tkd_conv = F.leaky_relu(self.tkd_conv(tkd))
            for i in range((block_number)):
                a_temp = self.decoder_unit[i](b0, mask)
                if i == 0:
                    x_temp = self.mod_conv1(tkd_conv, a_temp) + output_list[i]
                else:
                    x_temp = self.mod_conv1(x_temp, a_temp) + output_list[i]
            x = self.conv(x_temp)
            return x

