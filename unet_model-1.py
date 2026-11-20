from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from torch.nn.parameter import Parameter
from model.GEblock import GEblock
import numpy as np
import random

disp = False  # 当 disp = True时，条件成立，会执行缩进块内的print('conv', x1.shape)


def gaussian_blur_mask(x, mask_ratio=0.5, kernel_range=(3, 7), sigma_range=(0.5, 2.0)):
    """
    针对多聚焦退化的高斯模糊掩码（无cv2依赖，纯torch实现）
    x: 输入特征 [B, C, H, W]
    mask_ratio: 掩码比例，默认0.5
    return: 掩码后特征x_masked，掩码mask
    """
    B, C, H, W = x.shape
    device = x.device

    # 1. 生成空间掩码：1=保留原特征，0=替换为模糊特征
    mask = torch.ones((H, W), device=device)
    num_mask = int(H * W * mask_ratio)
    mask_flat = mask.flatten()
    mask_idx = torch.randperm(len(mask_flat), device=device)[:num_mask]
    mask_flat[mask_idx] = 0
    mask = mask_flat.reshape(H, W)

    # 2. 对掩码区域做高斯模糊


    # 只做一次模糊（减少计算）
    k = random.randint(kernel_range[0], kernel_range[1])
    if k % 2 == 0:
        k += 1
    sigma = random.uniform(sigma_range[0], sigma_range[1])

    # 生成高斯核（1D卷积核，只在高度方向模糊）
    kernel = torch.arange(k, device=device) - k // 2
    kernel = torch.exp(-kernel ** 2 / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()

    # 关键修复：使用2D卷积，确保输出尺寸与输入一致
    # 创建2D高斯核 [k, k]
    kernel_2d = kernel.view(k, 1) * kernel.view(1, k)  # 外积得到2D高斯核
    kernel_2d = kernel_2d.view(1, 1, k, k).repeat(C, 1, 1, 1)  # [C, 1, k, k]

    # 使用padding确保输出尺寸与输入一致
    pad = k // 2
    padded_x = F.pad(x, (pad, pad, pad, pad), mode='reflect')
    blurred = F.conv2d(padded_x, kernel_2d, groups=C)

    # 确保blurred的尺寸与x一致（以防万一）
    if blurred.shape[2:] != x.shape[2:]:
        blurred = blurred[:, :, :H, :W]

    # 将二维mask扩展为四维，匹配x_masked的维度
    mask_expanded = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    mask_expanded = mask_expanded.expand(B, C, H, W)  # [B, C, H, W]

    # 使用布尔索引赋值
    x_masked = torch.where(mask_expanded == 0, blurred, x)

    return x_masked, mask

# 通过可学习权重rho对IN和LN的结果进行加权求和，让网络在训练过程中自行决定在每个通道上，应该更偏向于使用 IN 还是 LN
# 但实际模型并没有调用此函数！！！
class ILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.0)
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

    def forward(self, input):
        # 实例归一化
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)

        # 层归一化
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)

        # rho 被扩展为 [B, C, 1, 1]，与 out_in 和 out_ln 形状匹配，输出 = 权重 * IN结果 + (1 - 权重) * LN结果
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln

        # 使用 gamma 和 beta 对混合后的归一化结果进行缩放和平移。
        out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + self.beta.expand(input.shape[0], -1, -1, -1)

        return out


# 层归一化，未被调用！！！
class ln(nn.Module):
    """
    Layer Normalization
    """
    def __init__(self, input):
        super(ln, self).__init__()
        self.ln = nn.LayerNorm(input.size()[1:]).cuda()
    def forward(self, x):
        x = self.ln(x)
        return x


# 卷积模块（特征提取两次）
class conv_block(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, in_ch, out_ch, down=False):
        super(conv_block, self).__init__()
        if down:             # 如果是下采样，步长为2
            stride = 2
        else:                # 如果不是下采样，步长为1
            stride = 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.SELU(inplace=False),

            # 在第一层卷积提取的基础上，再次提取（精炼或增强）
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.SELU(inplace=False))

    def forward(self, x):
        x = self.conv(x)
        return x

class Conv_block(nn.Module):
    """
    Global-Feature Encoding U-Net Convolution Block
    """

    # 输入通道，输出通道，卷积核，步长，循环次数/迭代深度，卷积填充模式：对称填充，使用实例归一化和selu激活函数。
    def __init__(self, in_ch, out_ch, kn_size, stride, t=2, pad_mode='SYMMETRIC', Normalize=True, Selu=True):
        super(Conv_block, self).__init__()
        self.pad_size = (kn_size - 1) // 2
        self.pad_mode = pad_mode
        self.Normalize = Normalize
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kn_size = kn_size
        self.stride = stride
        self.Selu = Selu
        self.t = t

        self.conv_block1 = self._make_layer1()
        self.conv_block2 = self._make_layer2()

    # 接收部分是self.in_ch，作用是接收外部传入的通道数，转换为可在内部使用的通道数（通道转换器）。
    def _make_layer1(self):
        layers = []

        # 如果需要填充，创建带有对称填充的卷积层；否则，创建不带有对称填充的，普通卷积层
        if self.pad_size > 0 and self.pad_mode is not None:
            layers.append(nn.Conv2d(self.in_ch, self.out_ch, kernel_size=self.kn_size, stride=self.stride, padding=self.pad_size, padding_mode=self.pad_mode))
        else:
            layers.append(nn.Conv2d(self.in_ch, self.out_ch, kernel_size=self.kn_size, stride=self.stride))

        # 归一化和激活函数（可选，可改为False）
        if self.Normalize == True:
            layers.append(nn.InstanceNorm2d(self.out_ch,affine=True,track_running_stats=True))
        if self.Selu == True:
            layers.append(nn.SELU(inplace=False))
        return nn.Sequential(*layers)

    # 接收layer1已经转换过的通道数out_ch，然后在layer2内部进行更深层次的特征提取。
    def _make_layer2(self):
        layers = []
        if self.pad_size > 0 and self.pad_mode is not None:
            layers.append(nn.Conv2d(self.out_ch, self.out_ch, kernel_size=self.kn_size, stride=self.stride, padding=self.pad_size, padding_mode=self.pad_mode))
        else:
            layers.append(nn.Conv2d(self.out_ch, self.out_ch, kernel_size=self.kn_size, stride=self.stride))
        if self.Normalize == True:
            layers.append(nn.InstanceNorm2d(self.out_ch,affine=True,track_running_stats=True))
        if self.Selu == True:
            layers.append(nn.SELU(inplace=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.t == 0 :   # 如果迭代数（提炼/循环次数为0，直接执行最简单的卷积快1，即layer1）
            x1 = self.conv_block1(x)
        else:              # 否则，遍历t，i=0即首次循环时，只经历block2得到X1，i=1时，将原始信号X+首次迭代得到的X1相加，再经过block2，赋值给X1
            for i in range(self.t):
                if i == 0:
                    x1 = self.conv_block2(x)
                x1 = self.conv_block2(x+x1)
        if disp:
            print('conv', x1.shape)
        return x1

class LightCalibHead(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=True),
            nn.BatchNorm2d(in_ch),
            nn.SELU(inplace=False),
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=True),
            nn.BatchNorm2d(in_ch)
        )
        # 初始化最后一层为0，使初始输出为恒等映射
        nn.init.zeros_(self.conv[3].weight)
        nn.init.zeros_(self.conv[3].bias)

    def forward(self, x):
        return self.conv(x) #只输出修正量


    # 上采样：恢复图像分辨率，扩大特征图空间尺寸并进行特征融合
class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(out_ch)
        self.In = nn.InstanceNorm2d(out_ch,affine=True,track_running_stats=True) # 没使用
        self.selu = nn.SELU(inplace=False) # 没使用！
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        # 双线性插值进行上采样，计算高效且不会引入棋盘伪影。
        x = F.interpolate(x, y.size()[2:], mode='bilinear', align_corners=True)
        z = self.conv(x)
        z = self.bn(z)
        z = self.relu(z)
        return z


# x是网络最终输出特征，y是两张原始图片
# 通过x的通道数，选择y的通道数从而使x+y输出，输出的是一个决策图，但还要经过其他处理才可变成最终决策图用于图像融合。
class res_layer(nn.Module):
    """
    Resblock layer
    """

    def __init__(self):
        super(res_layer, self).__init__()

    def forward(self, x, y, axis):
        if axis is not None:
            l = [y[:, i, :, :] for i in axis]
            y = torch.stack(l, 1)
        tensor = torch.add(x, y)
        return tensor


# 将压缩的全局特征向量（通常是全局池化或注意力机制得到的）扩展并复制到整个空间尺寸，
# 然后与局部空间特征图在通道维度上进行拼接。这样可以让每个空间位置都能"感知"到全局上下文信息。
# 全局特征金字塔提取模块的核心组件之一
class Global_concat_layer(nn.Module):
    """
    Global concat layer
    """
    def __init__(self):
        super(Global_concat_layer, self).__init__()

    # a：输入张量，dim：在哪个维度上重复，n_tile：重复几次
    # 先简单重复-构建交错索引-对重复后的张量重新排序
    # （A,B)-(A,B,A,B)-(A,A,B,B)
    def tile(self, a, dim, n_tile):
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
        return torch.index_select(a, dim, order_index.cuda())

    # x：局部空间特征图，y：全局特征
    def forward(self, x, y):
        batch_size = x.size()[0]   # 批次大小
        h = x.size()[2]            # 特征图高度
        w = x.size()[3]            # 特征图宽度

        # 假设输入Y的形状为[batch_size=2, channels=256, height=1, width=1]
        # 即执行前y_shape = [2,256,1,1],执行后y_shape = [2,256]
        concat_t = torch.squeeze(y, dim=2)  # 去掉索引为2的维度，索引从0开始，y_shape = [2,256,1]
        concat_t = torch.squeeze(concat_t, dim=2)  # 去掉索引为2的维度，索引从0开始，y_shape = [2,256]

        # 选取concat_t最后的一个张量赋给dims，然后重新规定concat_t的形状，确保形状无意外发生
        dims = concat_t.size()[-1]
        concat_t = torch.reshape(concat_t, [batch_size, dims])

        # 全局特征广播与拼接：通过遍历批次、交错重复和重塑，巧妙地将每个样本的一维全局特征向量转换成了与局部特征图空间尺寸相匹配的“常数平面”，
        # 然后通过通道拼接，使网络在每一个空间位置进行决策时，都能同时利用局部细节和全局上下文信息。
        bs = []
        for i in range(batch_size):
            batch = concat_t[i, :]
            batch = self.tile(batch, 0, h * w)
            batch = torch.reshape(batch, [-1, h, w])
            bs.append(batch)
        concat_t = torch.stack(bs)
        tensor = torch.cat([x, concat_t], 1)
        return tensor


# 从骨干网络提取的深层特征图中，通过并行与级联的多分支结构，提取并融合不同空间尺度（感受野大小）的上下文信息，生成一个富含全局语义的特征表示
# 全局特征金字塔提取模块核心组件之一
class Global_extract_layer(nn.Module):
    def __init__(self, channels=2048):
        """
        Global Feature Pyramid Extraction Module
        :type channels: int
        """
        super(Global_extract_layer, self).__init__()
        channels_mid = int(channels)

        self.channels_cond = channels

        # Master branch
        self.conv_master = nn.Conv2d(self.channels_cond, channels, kernel_size=3, bias=False)
        self.bn_master = nn.BatchNorm2d(channels)
        self.conv_master2 = nn.Conv2d(self.channels_cond * 2, self.channels_cond, kernel_size=3,bias=False)
        self.bn_master2 = nn.BatchNorm2d(channels)

        self.conv7x7_1 = nn.Conv2d(self.channels_cond, channels_mid, kernel_size=(7, 7), stride=2, padding=3, bias=False)
        self.bn1_1 = nn.BatchNorm2d(channels_mid)
        self.conv5x5_1 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(5, 5), stride=2, padding=2, bias=False)
        self.bn2_1 = nn.BatchNorm2d(channels_mid)
        self.conv3x3_1 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(3, 3), stride=2, padding=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(channels_mid)

        self.global_concat_layer = Global_concat_layer()
        self.conv7x7_2 = GEblock(channels_mid, 16,'att')
        self.conv5x5_2 = GEblock(channels_mid, 16,'att')
        self.conv3x3_2 = GEblock(channels_mid, 16,'att')

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        :param x: Shape: [b, 2048, h, w]
        :return: out: Feature maps. Shape: [b, 2048, h, w]
        """
        # Master branch
        x_master = self.conv_master(x)
        x_master = self.bn_master(x_master)

        # Branch 1
        x1_1 = self.conv3x3_1(x)
        x1_1 = self.bn3_1(x1_1)
        x1_1 = self.relu(x1_1)
        x1_2 = self.conv3x3_2(x1_1)

        # Branch 2
        x2_1 = self.conv5x5_1(x1_1)
        x2_1 = self.bn2_1(x2_1)
        x2_1 = self.relu(x2_1)
        x2_2 = self.conv5x5_2(x2_1)

        # Branch 3
        x3_1 = self.conv7x7_1(x2_1)
        x3_1 = self.bn1_1(x3_1)
        x3_1 = self.relu(x3_1)
        x3_2 = self.conv7x7_2(x3_1)

        # Merge branch 1 and 2
        g1 = x1_2 + x2_2 + x3_2
        g2 = self.global_concat_layer(x_master,g1)
        g3 = self.conv_master2(g2)
        g3 = self.bn_master2(g3)
        out = self.relu(g3)

        return out


# 基于全局注意力机制融合高层语义特征和低层细节特征
class GAU(nn.Module):
    def __init__(self, channels_high, channels_low, upsample=True):
        super(GAU, self).__init__()
        """
        Global Attention Connection Upsample module
        """
        self.upsample = upsample
        self.conv3x3 = nn.Conv2d(channels_low, channels_low, kernel_size=3, padding=1, bias=False)
        self.bn_low = nn.BatchNorm2d(channels_low)

        self.conv1x1 = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
        self.bn_high = nn.BatchNorm2d(channels_low)

        if upsample:
            self.conv_upsample = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
            self.bn_upsample = nn.BatchNorm2d(channels_low)

            self.conv_upsample2 = nn.Conv2d(channels_low*2, channels_low, kernel_size=1, padding=0, bias=False)
            self.bn_upsample2 = nn.BatchNorm2d(channels_low)
        else:
            self.conv_reduction = nn.Conv2d(channels_high, channels_low, kernel_size=3, padding=1, bias=False)
            self.bn_reduction = nn.BatchNorm2d(channels_low)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, fms_high, fms_low):
        """
        :param fms_high: Features of high level. Tensor.
        :param fms_low: Features of low level.  Tensor.
        :return: fms_att_upsample
        """
        b, c, h, w = fms_high.shape

        # 生成全局上下文向量（来自高层特征）
        fms_high_gp = nn.AvgPool2d(fms_high.shape[2:])(fms_high).view(len(fms_high), c, 1, 1)
        fms_high_gp = self.conv1x1(fms_high_gp)
        fms_high_gp = self.bn_high(fms_high_gp)
        fms_high_gp = self.relu(fms_high_gp)

        # 生成细节特征掩码（来自低层特征）
        fms_low_mask = self.conv3x3(fms_low)
        fms_low_mask = self.bn_low(fms_low_mask)

        # 生成空间注意力图
        fms_att = fms_low_mask * fms_high_gp
        if self.upsample:    # 特征融合与上采样
            high = self.relu(
                self.bn_upsample(self.conv_upsample(nn.Upsample(size=fms_att.size()[2:],mode='bilinear',align_corners=True)(fms_high))))
            out = torch.cat((high, fms_att), dim=1)
            out = self.relu(self.bn_upsample2(self.conv_upsample2(out)))
        else:
            out = self.relu(
                self.bn_reduction(self.conv_reduction(fms_high)) + fms_att)
        return out


# 上采样中转置卷积改为双线性插值+卷积；基础卷积块中使用BN和selu激活函数
class U_Net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=3, out_ch=1):
        super(U_Net, self).__init__()
        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5,e4)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5,e3)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4,e2)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3,e1)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        # d1 = self.active(out)

        return out

# 横向深度，全局特征金字塔提取模块，全局注意力上采样+残差模块
class GEU_Net(nn.Module):
    """
    The proposed GEU-Net network
    """
    def __init__(self, in_ch=3, out_ch=1):
        super(GEU_Net, self).__init__()

        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1],down=True)
        self.Conv3 = conv_block(filters[1], filters[2],down=True)
        self.Conv4 = conv_block(filters[2], filters[3],down=True)
        self.Conv5 = conv_block(filters[3], filters[4],down=True)

        self.GE_layer = Global_extract_layer(filters[4])

        # 新增：轻量特征校准头（仅训练时启用）
        self.calib_head = LightCalibHead(filters[4])  # filters[4]是Conv5的输出通道256

        self.gau_block1 = GAU(filters[4], filters[3])
        self.gau_block2 = GAU(filters[3], filters[2])
        self.gau_block3 = GAU(filters[2], filters[1])
        self.gau_block4 = GAU(filters[1], filters[0])
        self.gau = [self.gau_block1, self.gau_block2, self.gau_block3, self.gau_block4]

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Out1 = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        self.Out2 = res_layer()

        # ====== 逐级融合重建模块 ======
        # 各层特征投影到固定通道（这里设为16）
        self.recon_proj_d5 = nn.Conv2d(filters[3], 16, kernel_size=1)  # d5: 128ch, 32x32 → 16ch
        self.recon_proj_d4 = nn.Conv2d(filters[2], 16, kernel_size=1)  # d4: 64ch, 64x64 → 16ch
        self.recon_proj_d3 = nn.Conv2d(filters[1], 16, kernel_size=1)  # d3: 32ch, 128x128 → 16ch
        self.recon_proj_d2 = nn.Conv2d(filters[0], 16, kernel_size=1)  # d2: 16ch, 256x256 → 16ch

        # 三级融合卷积（输入均为32通道，输出16通道；最后一级输出1通道）
        self.recon_fuse_54 = nn.Conv2d(32, 16, kernel_size=3, padding=1)  # 融合 p5↑ + p4
        self.recon_fuse_43 = nn.Conv2d(32, 16, kernel_size=3, padding=1)  # 融合 fuse_54↑ + p3
        self.recon_fuse_32 = nn.Conv2d(32, 1, kernel_size=3, padding=1)  # 融合 fuse_43↑ + p2 → 最终单通道
        # =================================
    # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        # Downsampling
        e1 = self.Conv1(x)
        e2 = self.Conv2(e1)
        e3 = self.Conv3(e2)
        e4 = self.Conv4(e3)
        e5 = self.Conv5(e4)

        #训练和推理都执行，但推理时不做遮挡
        if self.training:
            x_masked, _ = gaussian_blur_mask(x, mask_ratio=0.5)
        else:
            x_masked = x  # 推理时直接使用原图，不做遮挡
        e5_masked = self.Conv1(x_masked)
        e5_masked = self.Conv2(e5_masked)
        e5_masked = self.Conv3(e5_masked)
        e5_masked = self.Conv4(e5_masked)
        e5_masked = self.Conv5(e5_masked)
        self.last_calib_output = self.calib_head(e5_masked)
        e5 = e5 + self.calib_head(e5_masked)


        # Extract global features
        g1 = self.GE_layer(e5)
        # Upsampling
        d5 = self.gau[0](g1, e4)
        x5 = self.Up5(g1, e4)
        d5 = torch.cat((x5, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.gau[1](d5, e3)
        x4 = self.Up4(d5, e3)
        d4 = torch.cat((d4, x4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.gau[2](d4, e2)
        x3 = self.Up3(d4, e2)
        d3 = torch.cat((d3, x3), dim=1)
        d3 = self.Up_conv3(d3)


        d2 = self.gau[3](d3, e1)
        x2 = self.Up2(d3, e1)
        d2 = torch.cat((d2, x2), dim=1)
        d2 = self.Up_conv2(d2)

        # ====== 逐级融合重建（拼接+卷积） ======
        # 1. 对各层级特征进行投影降维
        p5 = self.recon_proj_d5(d5)  # [B,16,32,32]
        p4 = self.recon_proj_d4(d4)  # [B,16,64,64]
        p3 = self.recon_proj_d3(d3)  # [B,16,128,128]
        p2 = self.recon_proj_d2(d2)  # [B,16,256,256]

        # 2. 从最深到最浅逐级融合（每次上采样2倍）
        # 第1级：p5 (32x32) → 上采样至64x64 → 与p4拼接 → 卷积融合
        p5_up = F.interpolate(p5, size=p4.shape[2:], mode='bilinear', align_corners=True)  # [B,16,64,64]
        fuse_54 = torch.cat([p5_up, p4], dim=1)  # [B,32,64,64]
        fuse_54 = self.recon_fuse_54(fuse_54)  # [B,16,64,64]

        # 第2级：fuse_54 (64x64) → 上采样至128x128 → 与p3拼接 → 卷积融合
        fuse_54_up = F.interpolate(fuse_54, size=p3.shape[2:], mode='bilinear', align_corners=True)  # [B,16,128,128]
        fuse_43 = torch.cat([fuse_54_up, p3], dim=1)  # [B,32,128,128]
        fuse_43 = self.recon_fuse_43(fuse_43)  # [B,16,128,128]

        # 第3级：fuse_43 (128x128) → 上采样至256x256 → 与p2拼接 → 卷积输出单通道
        fuse_43_up = F.interpolate(fuse_43, size=p2.shape[2:], mode='bilinear', align_corners=True)  # [B,16,256,256]
        fuse_32 = torch.cat([fuse_43_up, p2], dim=1)  # [B,32,256,256]
        recon_final = self.recon_fuse_32(fuse_32)  # [B,1,256,256]
        # ===========================================

        out = self.Out1(d2)
        out = self.Out2(out,x,[0,1])

        # d1 = self.active(out)

        return out, d2, recon_final


# 循环残差卷积；并没有被网络模型调用！！！
class RRCNN_block(nn.Module):
    """
    Recurrent Residual Convolutional Neural Network Block
    """
    def __init__(self, in_ch, out_ch, t=2, down=False):
        super(RRCNN_block, self).__init__()
        if down:
            stride = 2
        else:
            stride = 1
        self.RCNN = nn.Sequential(
            Conv_block(out_ch,out_ch, kn_size=3,stride=1,t=t),
            Conv_block(out_ch,out_ch, kn_size=3,stride=1,t=t)
        )
        self.conv1 = Conv_block(in_ch,out_ch, kn_size=1,stride=stride,t=0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.RCNN(x1)
        out = x1 + x2
        return out

# 纵向深度； 注意力与特征组合改进；
class R2U_Net(nn.Module):
    """
    R2U-Unet implementation
    Paper: https://arxiv.org/abs/1802.06955
    """

    def __init__(self, img_ch=3, output_ch=1, t=2):
        super(R2U_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(img_ch, filters[0], t=t)

        self.RRCNN2 = RRCNN_block(filters[0], filters[1], t=t)

        self.RRCNN3 = RRCNN_block(filters[1], filters[2], t=t)

        self.RRCNN4 = RRCNN_block(filters[2], filters[3], t=t)

        self.RRCNN5 = RRCNN_block(filters[3], filters[4], t=t)

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_RRCNN5 = RRCNN_block(filters[4], filters[3], t=t)

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_RRCNN4 = RRCNN_block(filters[3], filters[2], t=t)

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_RRCNN3 = RRCNN_block(filters[2], filters[1], t=t)

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_RRCNN2 = RRCNN_block(filters[1], filters[0], t=t)

        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

    # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.RRCNN1(x)

        e2 = self.Maxpool(e1)
        e2 = self.RRCNN2(e2)

        e3 = self.Maxpool1(e2)
        e3 = self.RRCNN3(e3)

        e4 = self.Maxpool2(e3)
        e4 = self.RRCNN4(e4)

        e5 = self.Maxpool3(e4)
        e5 = self.RRCNN5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        out = self.Conv(d2)

        # out = self.active(out)

        return out


# 注意力模块
class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(F_int,affine=True,track_running_stats=True)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(F_int,affine=True,track_running_stats=True)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(1,affine=True,track_running_stats=True),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)  # 处理深层特征
        x1 = self.W_x(x)  # 处理浅层特征
        psi = self.relu(g1 + x1)  # 特征相加并激活
        psi = self.psi(psi)  # 生成空间注意力权重图
        out = x * psi  # 原始浅层特征图与权重相乘
        return out

# 注意力聚焦
class AttU_Net(nn.Module):
    """
    Attention Unet implementation
    Paper: https://arxiv.org/abs/1804.03999
    """

    def __init__(self, img_ch=3, output_ch=1):
        super(AttU_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(img_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

        # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        # print(x5.shape)
        d5 = self.Up5(e5,e4)
        # print(d5.shape)
        x4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5,e3)
        x3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4,e2)
        x2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3,e1)
        x1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        #  out = self.active(out)

        return out

# 特征精炼
class R2AttU_Net(nn.Module):
    """
    Residual Recuurent Block with attention Unet
    Implementation : https://github.com/LeeJunHyun/Image_Segmentation
    """

    def __init__(self, in_ch=3, out_ch=1, t=2):
        super(R2AttU_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.RRCNN1 = RRCNN_block(in_ch, filters[0], t=t)
        self.RRCNN2 = RRCNN_block(filters[0], filters[1], t=t, down=True)
        self.RRCNN3 = RRCNN_block(filters[1], filters[2], t=t, down=True)
        self.RRCNN4 = RRCNN_block(filters[2], filters[3], t=t, down=True)
        self.RRCNN5 = RRCNN_block(filters[3], filters[4], t=t, down=True)

        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_RRCNN5 = RRCNN_block(filters[4], filters[3], t=t)

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_RRCNN4 = RRCNN_block(filters[3], filters[2], t=t)

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_RRCNN3 = RRCNN_block(filters[2], filters[1], t=t)

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_RRCNN2 = RRCNN_block(filters[1], filters[0], t=t)

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.RRCNN1(x)

        e2 = self.RRCNN2(e1)

        e3 = self.RRCNN3(e2)

        e4 = self.RRCNN4(e3)

        e5 = self.RRCNN5(e4)

        d5 = self.Up5(e5)
        e4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        e3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        e2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        e1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        out = self.Conv(d2)

        #  out = self.active(out)

        return out
