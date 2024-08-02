import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math
import random
import functools
import operator
import numpy as np










#################


class Attention(nn.Module):
    def __init__(self,
                 channels,
                 num_heads,
                 proj_drop=0.0,
                 kernel_size=3,
                 stride_kv=1,
                 stride_q=1,
                 padding_kv="same",
                 padding_q="same",
                 attention_bias=True
                 ):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.num_heads = num_heads
        self.proj_drop = proj_drop
        
        self.conv_q = nn.Conv2d(channels, channels, kernel_size, stride_q, padding_q, bias=attention_bias, groups=channels)
        self.layernorm_q = nn.LayerNorm(channels, eps=1e-5)
        self.conv_k = nn.Conv2d(channels, channels, kernel_size, stride_kv, stride_kv, bias=attention_bias, groups=channels)
        self.layernorm_k = nn.LayerNorm(channels, eps=1e-5)
        self.conv_v = nn.Conv2d(channels, channels, kernel_size, stride_kv, stride_kv, bias=attention_bias, groups=channels)
        self.layernorm_v = nn.LayerNorm(channels, eps=1e-5)
        
        self.attention = nn.MultiheadAttention(embed_dim=channels, 
                                               bias=attention_bias, 
                                               batch_first=True,
                                               # dropout = 0.0,
                                               num_heads=1)#num_heads=self.num_heads)

    def _build_projection(self, x, qkv):

        
        if qkv == "q":
            x1 = F.relu(self.conv_q(x))
            x1 = x1.permute(0, 2, 3, 1)
            x1 = self.layernorm_q(x1)
            proj = x1.permute(0, 3, 1, 2)
        elif qkv == "k":
            x1 = F.relu(self.conv_k(x))
            x1 = x1.permute(0, 2, 3, 1)
            x1 = self.layernorm_k(x1)
            proj = x1.permute(0, 3, 1, 2)            
        elif qkv == "v":
            x1 = F.relu(self.conv_v(x))
            x1 = x1.permute(0, 2, 3, 1)
            x1 = self.layernorm_v(x1)
            proj = x1.permute(0, 3, 1, 2)        

        return proj

    def forward_conv(self, x):
        q = self._build_projection(x, "q")
        k = self._build_projection(x, "k")
        v = self._build_projection(x, "v")

        return q, k, v

    def forward(self, x):
        q, k, v = self.forward_conv(x)
        q = q.view(x.shape[0], x.shape[1], x.shape[2]*x.shape[3])
        k = k.view(x.shape[0], x.shape[1], x.shape[2]*x.shape[3])
        v = v.view(x.shape[0], x.shape[1], x.shape[2]*x.shape[3])
        q = q.permute(0, 2, 1)
        k = k.permute(0, 2, 1)
        v = v.permute(0, 2, 1)
        x1 = self.attention(query=q, value=v, key=k, need_weights=False)
        
        x1 = x1[0].permute(0, 2, 1)
        x1 = x1.view(x1.shape[0], x1.shape[1], np.sqrt(x1.shape[2]).astype(int), np.sqrt(x1.shape[2]).astype(int))
        x1 = F.dropout(x1, self.proj_drop)

        return x1
 
class Transformer(nn.Module):

    def __init__(self,
                 # in_channels,
                 out_channels,
                 num_heads,
                 dpr,
                 proj_drop=0.0,
                 attention_bias=True,
                 padding_q="same",
                 padding_kv="same",
                 stride_kv=1,
                 stride_q=1):
        super().__init__()
        
        self.attention_output = Attention(channels=out_channels,
                                         num_heads=num_heads,
                                         proj_drop=proj_drop,
                                         padding_q=padding_q,
                                         padding_kv=padding_kv,
                                         stride_kv=stride_kv,
                                         stride_q=stride_q,
                                         attention_bias=attention_bias,
                                         )

        self.conv1 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
        self.layernorm = nn.LayerNorm(self.conv1.out_channels, eps=1e-5)
        self.wide_focus = Wide_Focus(out_channels, out_channels)

    def forward(self, x):
        x1 = self.attention_output(x)
        x1 = self.conv1(x1)
        x2 = torch.add(x1, x)
        #x3 = x2.permute(0, 2, 3, 1)
        #x3 = self.layernorm(x3)
        #x3 = x3.permute(0, 3, 1, 2)
        #x3 = self.wide_focus(x3)
        #x3_resized = F.interpolate(x3, size=(x2.shape[2], x2.shape[3]), mode='bilinear', align_corners=False)
        #x3 = torch.add(x2, x3_resized)
        return x2


class Wide_Focus(nn.Module): 
    """
    Wide-Focus module.
    """
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 4, 2, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 4, 2, padding=1, dilation=2)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 4, 2, padding=1, dilation=3)
        self.conv4 = nn.Conv2d(in_channels, out_channels, 4, 2, padding=1)


    def forward(self, x):
        x1 = self.conv1(x)
        x1 = F.gelu(x1)
        x1 = F.dropout(x1, 0.1)
        x2 = self.conv2(x)
        x2 = F.gelu(x2)
        x2 = F.dropout(x2, 0.1)
        x3 = self.conv3(x)
        x3 = F.gelu(x3)
        x3 = F.dropout(x3, 0.1)
        x1_resized = F.interpolate(x1, size=(x2.shape[2], x2.shape[3]), mode='bilinear', align_corners=False)
        added = torch.add(x1_resized, x2)
        x_out = self.conv4(added)
        x_out = F.gelu(x_out)
        x_out = F.dropout(x_out, 0.1)
        return x_out

####################

def conv(in_channels, out_channels, kernel_size, bias=False, padding = 1, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
        


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, bias=True, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()

        if bias:
            self.bias = nn.Parameter(torch.zeros(channel))

        else:
            self.bias = None

        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


def fused_leaky_relu(input, bias=None, negative_slope=0.2, scale=2 ** 0.5):
    if input.device.type == "cpu":
        if bias is not None:
            rest_dim = [1] * (input.ndim - bias.ndim - 1)
            return (
                F.leaky_relu(
                    input + bias.view(1, bias.shape[0], *rest_dim), negative_slope=0.2
                )
                * scale
            )

        else:
            return F.leaky_relu(input, negative_slope=0.2) * scale

    else:
        return FusedLeakyReLUFunction.apply(
            input.contiguous(), bias, negative_slope, scale
        )
        
##############################

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

##############################        

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class spatial_attn_layer(nn.Module):
    def __init__(self, kernel_size=5):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


    
    
class ca_layer(nn.Module):
    def __init__(self, channel, reduction=8, bias=True):
        super(ca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
    
class SCA(nn.Module):
    def __init__(
        self, n_feat, kernel_size=3, reduction=8,
        bias=False, bn=False, act=nn.PReLU(), res_scale=1):

        super(SCA, self).__init__()
        modules_body = [conv(n_feat, n_feat, kernel_size, bias=bias), act, conv(n_feat, n_feat, kernel_size, bias=bias)]
        self.body = nn.Sequential(*modules_body)
        
        ## Spatial Attention
        self.SA = spatial_attn_layer()

        ## Channel Attention        
        self.CA = ca_layer(n_feat,reduction, bias=bias)

        self.conv1x1 = nn.Conv2d(n_feat*2, n_feat, kernel_size=1, bias=bias)

    def forward(self, x):
        res = self.body(x)
        sa_branch = self.SA(res)
        ca_branch = self.CA(res)
        res = torch.cat([sa_branch, ca_branch], dim=1)
        res = self.conv1x1(res)
        res += x
        return res



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return x + self.conv_block(x)

class _GradientScalarLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.weight = weight
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return ctx.weight*grad_input, None

gradient_scalar = _GradientScalarLayer.apply


class GradientScalarLayer(torch.nn.Module):
    def __init__(self, weight=4):
        super(GradientScalarLayer, self).__init__()
        self.weight = weight

    def forward(self, input):
        return gradient_scalar(input, self.weight)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "weight=" + str(self.weight)
        tmpstr += ")"
        return tmpstr











class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, bn=True):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if bn:
            layers.append(nn.BatchNorm2d(out_size, momentum=0.8))
        layers.append(nn.LeakyReLU(0.2))
        #layers.append(GradientScalarLayer())
        layers.append(ResidualBlock(out_size, out_size))  # Add residual block
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)




class UNetDown1(nn.Module):
    def __init__(self, in_size, out_size, bn=True):
        super(UNetDown1, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if bn:
            layers.append(nn.BatchNorm2d(out_size, momentum=0.8))
        layers.append(nn.LeakyReLU(0.2))
        #layers.append(GradientScalarLayer())
        #layers.append(ResidualBlock(out_size, out_size))  # Add residual block
        layers.append(Transformer(out_channels=out_size,num_heads=2,dpr=4))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
 





class MutilScal(nn.Module):
    def __init__(self, dim=512, fc_ratio=4, dilation=[3, 5, 7], pool_ratio=16):
        super(MutilScal, self).__init__()
        self.conv0_1 = nn.Conv2d(dim, dim//fc_ratio, 1)
        self.bn0_1 = nn.BatchNorm2d(dim//fc_ratio)
        self.conv0_2 = nn.Conv2d(dim//fc_ratio, dim//fc_ratio, 3, padding=dilation[-3], dilation=dilation[-3], groups=dim //fc_ratio)
        self.bn0_2 = nn.BatchNorm2d(dim // fc_ratio)
        self.conv0_3 = nn.Conv2d(dim//fc_ratio, dim, 1)
        self.bn0_3 = nn.BatchNorm2d(dim)

        self.conv1_2 = nn.Conv2d(dim//fc_ratio, dim//fc_ratio, 3, padding=dilation[-2], dilation=dilation[-2], groups=dim // fc_ratio)
        self.bn1_2 = nn.BatchNorm2d(dim//fc_ratio)
        self.conv1_3 = nn.Conv2d(dim//fc_ratio, dim, 1)
        self.bn1_3 = nn.BatchNorm2d(dim)

        self.conv2_2 = nn.Conv2d(dim//fc_ratio, dim//fc_ratio, 3, padding=dilation[-1], dilation=dilation[-1], groups=dim//fc_ratio)
        self.bn2_2 = nn.BatchNorm2d(dim//fc_ratio)
        self.conv2_3 = nn.Conv2d(dim//fc_ratio, dim, 1)
        self.bn2_3 = nn.BatchNorm2d(dim)

        self.conv3 = nn.Conv2d(dim, dim, 1)
        self.bn3 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU6()

        self.Avg = nn.AdaptiveAvgPool2d(pool_ratio)

    def forward(self, x):
        u = x.clone()

        attn0_1 = self.relu(self.bn0_1(self.conv0_1(x)))
        attn0_2 = self.relu(self.bn0_2(self.conv0_2(attn0_1)))
        attn0_3 = self.relu(self.bn0_3(self.conv0_3(attn0_2)))

        attn1_2 = self.relu(self.bn1_2(self.conv1_2(attn0_1)))
        attn1_3 = self.relu(self.bn1_3(self.conv1_3(attn1_2)))

        attn2_2 = self.relu(self.bn2_2(self.conv2_2(attn0_1)))
        attn2_3 = self.relu(self.bn2_3(self.conv2_3(attn2_2)))

        attn = attn0_3 + attn1_3 + attn2_3
        attn = self.relu(self.bn3(self.conv3(attn)))
        attn = attn * u

        pool = self.Avg(attn)

        return pool
    
class Mutilscal_MHSA(nn.Module):
    def __init__(self, dim, num_heads, atten_drop = 0., proj_drop = 0., dilation = [3, 5, 7], fc_ratio=4, pool_ratio=16):
        super(Mutilscal_MHSA, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.atten_drop = nn.Dropout(atten_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.MSC = MutilScal(dim=dim, fc_ratio=fc_ratio, dilation=dilation, pool_ratio=pool_ratio)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim//fc_ratio, kernel_size=1),
            nn.ReLU6(),
            nn.Conv2d(in_channels=dim//fc_ratio, out_channels=dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.kv = conv(dim, 2 * dim, 1)

    def forward(self, x):
        u = x.clone()
        B, C, H, W = x.shape
        kv = self.MSC(x)
        kv = self.kv(kv)

        B1, C1, H1, W1 = kv.shape

        q = rearrange(x, 'b (h d) (hh) (ww) -> (b) h (hh ww) d', h=self.num_heads,
                      d=C // self.num_heads, hh=H, ww=W)
        k, v = rearrange(kv, 'b (kv h d) (hh) (ww) -> kv (b) h (hh ww) d', h=self.num_heads,
                         d=C // self.num_heads, hh=H1, ww=W1, kv=2)

        dots = (q @ k.transpose(-2, -1)) * self.scale
        attn = dots.softmax(dim=-1)
        attn = self.atten_drop(attn)
        attn = attn @ v

        attn = rearrange(attn, '(b) h (hh ww) d -> b (h d) (hh) (ww)', h=self.num_heads,
                         d=C // self.num_heads, hh=H, ww=W)
        c_attn = self.avgpool(x)
        c_attn = self.fc(c_attn)
        c_attn = c_attn * u
        return attn + c_attn
    
    
    
    
class UNetUp2(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_size, momentum=0.8),
            nn.ReLU(inplace=True),
        ]
        layers.append(ResidualBlock(out_size, out_size))  # Add residual block
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

        



    def forward(self, x1, x2):
        
        ###
        cat_fea = torch.cat([x1,x2], dim=1)
        
        ###
        att_vec_1  = self.gate_1(cat_fea)
        att_vec_2  = self.gate_2(cat_fea)

        att_vec_cat  = torch.cat([att_vec_1, att_vec_2], dim=1)
        att_vec_soft = self.softmax(att_vec_cat)
        
        att_soft_1, att_soft_2 = att_vec_soft[:, 0:1, :, :], att_vec_soft[:, 1:2, :, :]
        x_fusion = x1 * att_soft_1 + x2 * att_soft_2
        
        return x_fusion



class CFF2(nn.Module):
    def __init__(self, in_channel1, in_channel2, out_channel):
        super(CFF, self).__init__()
        
        act_fn = nn.ReLU(inplace=True)
        
        # Define layers for processing features
        self.layer0 = BasicConv(in_channel1, out_channel // 2, 1)
        self.layer1 = BasicConv(in_channel2, out_channel // 2, 1)
        
        self.layer3_1 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel // 2),
            act_fn
        )
        self.layer3_2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel // 2),
            act_fn
        )
        self.layer5_1 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel // 2, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(out_channel // 2),
            act_fn
        )
        self.layer5_2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel // 2, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(out_channel // 2),
            act_fn
        )
        self.layer_out = nn.Sequential(
            nn.Conv2d(out_channel // 2, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            act_fn
        )

    def forward(self, x0, x1):
        x0_1 = self.layer0(x0)
        x1_1 = self.layer1(x1)
        
        # Check dimensions of tensors
        print("x0_1 shape:", x0_1.shape)
        print("x1_1 shape:", x1_1.shape)
        
        x1_1_downsampled = F.interpolate(x1_1, size=x0_1.shape[2:], mode='bilinear', align_corners=False)

        # Concatenate features and process them
        x_concat = torch.cat((x0_1, x1_1_downsampled), dim=1)
        x_3_1 = self.layer3_1(x_concat)
        x_5_1 = self.layer5_1(x_concat)
        
        x_concat2 = torch.cat((x_3_1, x_5_1), dim=1)
        x_3_2 = self.layer3_2(x_concat2)
        x_5_2 = self.layer5_2(x_concat2)
        
        # Final output
        out = self.layer_out(x0_1 + x1_1 + x_3_2 * x_5_2)
        
        return out

import torch.nn.functional as F

class SelfSimilarityEncoder(nn.Module):
    def __init__(self, in_ch, mid_ch, unfold_size, ksize):
        super(SelfSimilarityEncoder, self).__init__()
            
        def make_building_conv_block(in_channel, out_channel, ksize, padding=(0,0,0), stride=(1,1,1), bias=True, conv_group=1):
            building_block_layers = []
            building_block_layers.append(nn.Conv3d(in_channel, out_channel, (1, ksize, ksize),
                                             stride=stride, bias=bias, groups=conv_group, padding=padding))
            building_block_layers.append(nn.BatchNorm3d(out_channel))
            building_block_layers.append(nn.ReLU(inplace=True))

            return nn.Sequential(*building_block_layers)

        conv_in_block_list = [make_building_conv_block(mid_ch, mid_ch, ksize) for _ in range(unfold_size//2)]
        self.conv_in = nn.Sequential(*conv_in_block_list)
        self.conv1x1_out = nn.Sequential(
            nn.Conv2d(mid_ch, in_ch, kernel_size=1, bias=True, padding=0),
            nn.BatchNorm2d(in_ch))

    def forward(self, x):
        b, c, h, w, u, v = x.shape

        x = x.view(b, c, h * w, u, v)
        x = self.conv_in(x)
        c = x.shape[1]
        x = x.mean(dim=[-1,-2]).view(b, c, h, w)
        x = self.conv1x1_out(x)  # [B, C3, H, W] -> [B, C4, H, W]

        return x

class SSM(nn.Module):
    def __init__(self, in_ch, mid_ch, unfold_size=7, ksize=3):
        super(SSM, self).__init__()
        
        self.ch_reduction_encoder = nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False, padding=0)
        self.SCC = SelfCorrelationComputation(unfold_size=unfold_size)
        self.SSE = SelfSimilarityEncoder(in_ch, mid_ch, unfold_size=unfold_size, ksize=ksize)

        self.FFN = nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=True, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=True, padding=0))
     
    def forward(self, ssm_input_feat):
        q = self.ch_reduction_encoder(ssm_input_feat)
        q = F.normalize(q, dim=1, p=2)
            
        self_sim = self.SCC(q)
        self_sim_feat = self.SSE(self_sim)
        ssm_output_feat = ssm_input_feat + self_sim_feat
        ssm_output_feat = self.FFN(ssm_output_feat)

        return ssm_output_feat

class SelfCorrelationComputation(nn.Module):
    def __init__(self, unfold_size=5):
        super(SelfCorrelationComputation, self).__init__()
        self.unfold_size = (unfold_size, unfold_size)
        self.padding_size = unfold_size // 2
        self.unfold = nn.Unfold(kernel_size=self.unfold_size, padding=self.padding_size)

    def forward(self, q):
        b, c, h, w = q.shape

        q_unfold = self.unfold(q)  # b, cuv, h, w
        q_unfold = q_unfold.view(b, c, self.unfold_size[0], self.unfold_size[1], h, w) # b, c, u, v, h, w
        self_sim = q_unfold * q.unsqueeze(2).unsqueeze(2)  # b, c, u, v, h, w * b, c, 1, 1, h, w
        self_sim = self_sim.permute(0, 1, 4, 5, 2, 3).contiguous()  # b, c, h, w, u, v

        return self_sim.clamp(min=0)


#########################################################################################


#############################################################################################


def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, relu=True, bn=False):
    conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding)

    out = [conv]

    if bn:
        out.append(nn.BatchNorm2d(out_channels, affine=False))
    if relu:
        out.append(nn.ReLU())

    return nn.Sequential(*out)


class SPPS(nn.Module):
    def __init__(self, in_channels, out_channels, use_se=False, relu=True, bn=True):
        super(SPPS, self).__init__()

        self.conv1 = conv(in_channels=in_channels, out_channels=out_channels , kernel_size=3, stride=2, padding=1,
                          relu=relu, bn=bn)
        self.conv2 = conv(in_channels=in_channels, out_channels=out_channels , kernel_size=3, stride=4, padding=1,
                          relu=relu, bn=bn)
        self.conv3 = conv(in_channels=in_channels, out_channels=out_channels , kernel_size=3, stride=8, padding=1,
                          relu=relu, bn=bn)

    def forward(self, x,x1):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x)

        x2_upsample = F.interpolate(x2, scale_factor=4., mode="bilinear", align_corners=False)
        x3_upsample = F.interpolate(x3, scale_factor=4., mode="bilinear", align_corners=False)

        print("x1: ", x1.shape)
        print("x2: ", x2.shape)
        # print("x3: ", x3.shape)
        print("x2_s: ", x2_upsample.shape)
        print("x3_s: ", x3_upsample.shape)

        return torch.cat([x1, x2_upsample, x3_upsample], dim=1)



##########################################################################################




#######################################################################################

class CFF(nn.Module):
    def __init__(self, in_channel1, in_channel2, out_channel):
        super(CFF, self).__init__()

        act_fn = nn.ReLU(inplace=True)

        ## ---------------------------------------- ##
        self.layer0 = BasicConv(in_channel1, out_channel // 2, 1)
        self.layer1 = BasicConv(in_channel2, out_channel // 2, 1)

        self.layer3_1 = nn.Sequential(nn.Conv2d(out_channel, out_channel // 2, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(out_channel // 2), act_fn)
        self.layer3_2 = nn.Sequential(nn.Conv2d(out_channel, out_channel // 2, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(out_channel // 2), act_fn)

        self.layer5_1 = nn.Sequential(nn.Conv2d(out_channel, out_channel // 2, kernel_size=5, stride=1, padding=2),
                                      nn.BatchNorm2d(out_channel // 2), act_fn)
        self.layer5_2 = nn.Sequential(nn.Conv2d(out_channel, out_channel // 2, kernel_size=5, stride=1, padding=2),
                                      nn.BatchNorm2d(out_channel // 2), act_fn)

        self.layer_out = nn.Sequential(nn.Conv2d(out_channel // 2, out_channel, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(out_channel), act_fn)
                                       
        #self.layerA = Attention1(in_channel1,in_channel2)
        

    def forward(self, x0, x1):
        ## ------------------------------------------------------------------ ##
        x0_1 = self.layer0(x0)
        x1_1_resized = F.interpolate(x1, size=x0_1.shape[2:], mode='bilinear', align_corners=False)
        
        x1_1 = self.layer1(x1_1_resized)

        # Concatenate features and process them
        x_concat = torch.cat((x0_1, x1_1), dim=1)
        x_3_1 = self.layer3_1(x_concat)
        x_5_1 = self.layer5_1(x_concat)
        
        
       # x_3_1 = self.layerA(x_3_1)
        #x_5_1 = self.layerA(x_5_1)

        x_concat2 = torch.cat((x_3_1, x_5_1), dim=1)
        x_3_2 = self.layer3_2(x_concat2)
        x_5_2 = self.layer5_2(x_concat2)

        out = self.layer_out(x0_1 + x1_1 + torch.mul(x_3_2, x_5_2))

        return out
        
##############################################################################################




##########################################################


class CFFnew2(nn.Module):
    def __init__(self, in_channel1, in_channel2, out_channel):
        super(CFFnew2, self).__init__()

        act_fn = nn.ReLU(inplace=True)

        ## ---------------------------------------- ##
        self.layer0 = BasicConv(in_channel1, out_channel // 2, 1)
        self.layer1 = BasicConv(in_channel2, out_channel // 2, 1)

        self.layer3_1 = nn.Sequential(nn.Conv2d(out_channel, out_channel // 2, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(out_channel // 2), act_fn)
        self.layer3_2 = nn.Sequential(nn.Conv2d(out_channel, out_channel // 2, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(out_channel // 2), act_fn)

        self.layer5_1 = nn.Sequential(nn.Conv2d(out_channel, out_channel // 2, kernel_size=5, stride=1, padding=2),
                                      nn.BatchNorm2d(out_channel // 2), act_fn)
        self.layer5_2 = nn.Sequential(nn.Conv2d(out_channel, out_channel // 2, kernel_size=5, stride=1, padding=2),
                                      nn.BatchNorm2d(out_channel // 2), act_fn)

        self.layer_out = nn.Sequential(nn.Conv2d(out_channel // 2, out_channel, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(out_channel), act_fn)
                                       
        self.layerA = CALayer(in_channel1)
        
        

    def forward(self, x0, x1):
        ## ------------------------------------------------------------------ ##
        x0_1 = self.layer0(x0)
        x1_1_resized = F.interpolate(x1, size=x0_1.shape[2:], mode='bilinear', align_corners=False)
        
        x1_1 = self.layer1(x1_1_resized)

        # Concatenate features and process them
        x_concat = torch.cat((x0_1, x1_1), dim=1)
        
        
        x_concat1 = self.layerA(x_concat)
          
        x_3_1 = self.layer3_1(x_concat1)
        x_5_1 = self.layer5_1(x_concat1)
        
        x_concat2 = torch.cat((x_3_1, x_5_1), dim=1)
        
        x_concat3 = self.layerA(x_concat2)
        
        x_3_2 = self.layer3_2(x_concat3)
        x_5_2 = self.layer5_2(x_concat3)

        out = self.layer_out(x0_1 + x1_1 + torch.mul(x_3_2, x_5_2))

        return out
##############################################################################################





#####################      



###############################################################

class RelativePositionBias(nn.Module):
    # input-independent relative position attention
    # As the number of parameters is smaller, so use 2D here
    # Borrowed some code from SwinTransformer: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
    def __init__(self, num_heads, h, w):
        super().__init__()
        self.num_heads = num_heads
        self.h = h
        self.w = w

        self.relative_position_bias_table = nn.Parameter(
                torch.randn((2*h-1) * (2*w-1), num_heads)*0.02)

        coords_h = torch.arange(self.h)
        coords_w = torch.arange(self.w)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w])) # 2, h, w
        coords_flatten = torch.flatten(coords, 1) # 2, hw

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.h - 1
        relative_coords[:, :, 1] += self.w - 1
        relative_coords[:, :, 0] *= 2 * self.h - 1
        relative_position_index = relative_coords.sum(-1) # hw, hw
    
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, H, W):
        
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.h, self.w, self.h*self.w, -1) #h, w, hw, nH
        relative_position_bias_expand_h = torch.repeat_interleave(relative_position_bias, H//self.h, dim=0)
        relative_position_bias_expanded = torch.repeat_interleave(relative_position_bias_expand_h, W//self.w, dim=1) #HW, hw, nH
        
        relative_position_bias_expanded = relative_position_bias_expanded.view(H*W, self.h*self.w, self.num_heads).permute(2, 0, 1).contiguous().unsqueeze(0)

        return relative_position_bias_expanded

class depthwise_separable_conv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch, bias=bias, stride=stride)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)

    def forward(self, x): 
        out = self.depthwise(x)
        out = self.pointwise(out)

        return out 

class LinearAttention(nn.Module):
    
    def __init__(self, dim, heads=4, dim_head=64, attn_drop=0., proj_drop=0., reduce_size=16, projection='interp', rel_pos=True):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** (-0.5)
        self.dim_head = dim_head
        self.reduce_size = reduce_size
        self.projection = projection
        self.rel_pos = rel_pos
        
        # depthwise conv is slightly better than conv1x1
        #self.to_qkv = nn.Conv2d(dim, self.inner_dim*3, kernel_size=1, stride=1, padding=0, bias=True)
        #self.to_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, stride=1, padding=0, bias=True)
       
        self.to_qkv = depthwise_separable_conv(dim, self.inner_dim*3)
        self.to_out = depthwise_separable_conv(self.inner_dim, dim)


        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.rel_pos:
            # 2D input-independent relative position encoding is a little bit better than
            # 1D input-denpendent counterpart
            self.relative_position_encoding = RelativePositionBias(heads, reduce_size, reduce_size)
            #self.relative_position_encoding = RelativePositionEmbedding(dim_head, reduce_size)

    def forward(self, x):

        B, C, H, W = x.shape

        #B, inner_dim, H, W
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        if self.projection == 'interp' and H != self.reduce_size:
            k, v = map(lambda t: F.interpolate(t, size=self.reduce_size, mode='bilinear', align_corners=True), (k, v))

        elif self.projection == 'maxpool' and H != self.reduce_size:
            k, v = map(lambda t: F.adaptive_max_pool2d(t, output_size=self.reduce_size), (k, v))
        
        q = rearrange(q, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head, heads=self.heads, h=H, w=W)
        k, v = map(lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head, heads=self.heads, h=self.reduce_size, w=self.reduce_size), (k, v))

        q_k_attn = torch.einsum('bhid,bhjd->bhij', q, k)
        
        if self.rel_pos:
            relative_position_bias = self.relative_position_encoding(H, W)
            q_k_attn += relative_position_bias
            #rel_attn_h, rel_attn_w = self.relative_position_encoding(q, self.heads, H, W, self.dim_head)
            #q_k_attn = q_k_attn + rel_attn_h + rel_attn_w

        q_k_attn *= self.scale
        q_k_attn = F.softmax(q_k_attn, dim=-1)
        q_k_attn = self.attn_drop(q_k_attn)

        out = torch.einsum('bhij,bhjd->bhid', q_k_attn, v)
        out = rearrange(out, 'b heads (h w) dim_head -> b (dim_head heads) h w', h=H, w=W, dim_head=self.dim_head, heads=self.heads)

        out = self.to_out(out)
        out = self.proj_drop(out)

        return out, q_k_attn


class BasicTransBlock(nn.Module):

    def __init__(self, in_ch, heads=2, dim_head=2, attn_drop=0., proj_drop=0., reduce_size=16, projection='interp', rel_pos=True):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)

        self.attn = LinearAttention(in_ch, heads=heads, dim_head=in_ch//heads, attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos)
        
        self.bn2 = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU(inplace=True)
        self.mlp = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False)
        # conv1x1 has not difference with mlp in performance

    def forward(self, x):

        out = self.bn1(x)
        out, q_k_attn = self.attn(out)
        
        out = out + x
        residue = out

        out = self.bn2(out)
        out = self.relu(out)
        out = self.mlp(out)

        out += residue

        return out

##############################################################
        
        
class UNetUp11(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUp11, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_size, momentum=0.8),
            nn.ReLU(inplace=True),
        ]
        
        layers.append(ResidualBlock(out_size, out_size))  # Add residual block
        self.model = nn.Sequential(*layers)
        

    def forward(self, x, skip_input1, skip_input2):
        x = self.model(x)
      
        x = torch.cat((x, skip_input1,skip_input2), 1)
        
        return x





class UNetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_size, momentum=0.8),
            nn.ReLU(inplace=True),
        ]
        #layers.append(GradientScalarLayer())
        layers.append(ResidualBlock(out_size, out_size))  # Add residual block
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


####################################################################################


class Wide_Focus(nn.Module): 
    """
    Wide-Focus module.
    """
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same", dilation=2)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same", dilation=3)
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")


    def forward(self, x):
        x1 = self.conv1(x)
        x1 = F.gelu(x1)
        x1 = F.dropout(x1, 0.1)
        x2 = self.conv2(x)
        x2 = F.gelu(x2)
        x2 = F.dropout(x2, 0.1)
        x3 = self.conv3(x)
        x3 = F.gelu(x3)
        x3 = F.dropout(x3, 0.1)
        added = torch.add(x1, x2)
        added = torch.add(added, x3)
        x_out = self.conv4(added)
        x_out = F.gelu(x_out)
        x_out = F.dropout(x_out, 0.1)
        return x_out
        


################################################################################



import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import pdb



def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)

class depthwise_separable_conv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch, bias=bias, stride=stride)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)

    def forward(self, x): 
        out = self.depthwise(x)
        out = self.pointwise(out)

        return out 

class Mlp(nn.Module):
    def __init__(self, in_ch, hid_ch=None, out_ch=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_ch = out_ch or in_ch
        hid_ch = hid_ch or in_ch

        self.fc1 = nn.Conv2d(in_ch, hid_ch, kernel_size=1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hid_ch, out_ch, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = nn.Sequential(
                    nn.BatchNorm2d(inplanes),
                    self.relu, 
                    nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
                    )

    def forward(self, x): 
        residue = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += self.shortcut(residue)

        return out

class BasicTransBlock(nn.Module):

    def __init__(self, in_ch, heads, attn_drop=0., proj_drop=0., reduce_size=16, projection='interp', rel_pos=True):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)

        self.attn = LinearAttention(in_ch, heads=heads, dim_head=in_ch//heads, attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos)
        
        self.bn2 = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU(inplace=True)
        self.mlp = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False)
        # conv1x1 has not difference with mlp in performance

    def forward(self, x):

        out = self.bn1(x)
        out, q_k_attn = self.attn(out)
        
        out = out + x
        residue = out

        out = self.bn2(out)
        out = self.relu(out)
        out = self.mlp(out)

        out += residue

        return out


class RelativePositionBias(nn.Module):
    # input-independent relative position attention
    # As the number of parameters is smaller, so use 2D here
    # Borrowed some code from SwinTransformer: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
    def __init__(self, num_heads, h, w):
        super().__init__()
        self.num_heads = num_heads
        self.h = h
        self.w = w

        self.relative_position_bias_table = nn.Parameter(
                torch.randn((2*h-1) * (2*w-1), num_heads)*0.02)

        coords_h = torch.arange(self.h)
        coords_w = torch.arange(self.w)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w])) # 2, h, w
        coords_flatten = torch.flatten(coords, 1) # 2, hw

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.h - 1
        relative_coords[:, :, 1] += self.w - 1
        relative_coords[:, :, 0] *= 2 * self.h - 1
        relative_position_index = relative_coords.sum(-1) # hw, hw
    
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, H, W):
        
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.h, self.w, self.h*self.w, -1) #h, w, hw, nH
        relative_position_bias_expand_h = torch.repeat_interleave(relative_position_bias, H//self.h, dim=0)
        relative_position_bias_expanded = torch.repeat_interleave(relative_position_bias_expand_h, W//self.w, dim=1) #HW, hw, nH
        
        relative_position_bias_expanded = relative_position_bias_expanded.view(H*W, self.h*self.w, self.num_heads).permute(2, 0, 1).contiguous().unsqueeze(0)

        return relative_position_bias_expanded

####################################################################################

class UNetDown2(nn.Module):
    def __init__(self, in_size, out_size, bn=False):
        super(UNetDown2, self).__init__()
        layers = []
        #if bn:
            #layers.append(nn.BatchNorm2d(out_size, momentum=0.8))
        #layers.append(nn.LeakyReLU(0.2))
        #layers.append(GradientScalarLayer())
        #layers.append(ResidualBlock(out_size, out_size))  # Add residual block
        layers.append( BasicTransBlock(in_ch=in_size, heads=2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


####################################################################################

class MuLA_GAN_Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(MuLA_GAN_Generator, self).__init__()
        # encoding layers
        self.down1 = UNetDown(in_channels, 32)   #, bn=False)
        self.down2 = UNetDown(32, 128)
        
        self.cff1 = CFF(128, 32, 128)  # Add CFF module between down2 and down3
        self.down3 = UNetDown(128, 256)
        
        self.cff2 = CFF(256, 128, 256)  # Add CFF module between down3 and down4
        self.down4 = UNetDown(256, 256)
        
        self.down5 = UNetDown(256, 256 )   #, bn=False)

        self.attention1 = Mutilscal_MHSA(32,4)
        self.attention2 = Mutilscal_MHSA(128,4)
        self.attention3 = Mutilscal_MHSA(256,4)
        self.attention4 = Mutilscal_MHSA(256,4)
        
        
        self.wdf1 = UNetDown2(32,128)
        self.wdf2 = UNetDown2(128,128)
        self.wdf3 = UNetDown2(256,256)
        self.wdf4 = UNetDown2(256,256)

        # decoding layers
        self.up1 = UNetUp11(256, 256)
        self.up2 = UNetUp11(768, 256)
        self.up3 = UNetUp11(768, 128)
        self.up4 = UNetUp11(384, 32)
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(96, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        
        d1initial = self.down1(x)
        d1 = self.attention1(d1initial)
        
        d2initial = self.down2(d1)
        d2 = self.attention2(d2initial)
        
        d3base = self.cff1(d2, d1)  # Pass output of CFF to down3
        d3initial = self.down3(d3base)
        d3 = self.attention3(d3initial)
        
        
        d4base = self.cff2(d3, d2)  # Pass output of CFF to down4
        d4initial = self.down4(d4base)
        d4 = self.attention4(d4initial)
        
        
        t1 = self.wdf1(d1initial)
        t2 = self.wdf2(d2initial)
        t3 = self.wdf3(d3initial)
        t4 = self.wdf4(d4initial)
        
        
        d5 = self.down5(d4)

        u1 = self.up1(d5, d4, t4)
        u2 = self.up2(u1, d3 ,t3)
        u3 = self.up3(u2, d2, t2)
        u4 = self.up4(u3, d1 , t1)

        return self.final(u4)


class Discriminator(nn.Module):
    """ A 4-layer Markovian discriminator
    """
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            #Returns downsampling layers of each discriminator block
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if bn: layers.append(nn.BatchNorm2d(out_filters, momentum=0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            #layers.append( SSM( in_filters, out_filters))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels*2, 32, bn=False),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(256, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)














