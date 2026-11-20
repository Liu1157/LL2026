import torch
from torch import nn
import torch.nn.init as init

# 为输入的特征图生成一个“使用说明书”或“调制指南”，告诉网络的后续部分：“在当前全局上下文中，你应该更关注输入特征的哪些通道”
# 输出的是用于调制原始特征通道权重的调整量，不是处理后的特征图！！
# 输入特征图 -> 空间池化（生成全局向量）-> 通道注意力网络（生成调整量）-> 输出调整量

# 初始化，替代mmcv
def constant_init(module, val, bias=0):
    """Constant initialization."""
    if hasattr(module, 'weight') and module.weight is not None:
        init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        init.constant_(module.bias, bias)

# 初始化，替代mmcv
def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    """Kaiming initialization."""
    if distribution == 'normal':
        if hasattr(module, 'weight') and module.weight is not None:
            init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        if hasattr(module, 'weight') and module.weight is not None:
            init.kaiming_uniform_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)

    if hasattr(module, 'bias') and module.bias is not None:
        init.constant_(module.bias, bias)


# 初始化，替代mmcv
def xavier_init(module, gain=1, bias=0, distribution='normal'):
    """Xavier initialization (compatibility wrapper)."""
    if distribution == 'normal':
        if hasattr(module, 'weight') and module.weight is not None:
            init.xavier_normal_(module.weight, gain=gain)
    else:
        if hasattr(module, 'weight') and module.weight is not None:
            init.xavier_uniform_(module.weight, gain=gain)

    if hasattr(module, 'bias') and module.bias is not None:
        init.constant_(module.bias, bias)




# 初始化，替代mmcv
def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
        m[-1].inited = True
    else:
        constant_init(m, val=0)
        m.inited = True


class GEblock(nn.Module):
    """
    Global Feature Extraction Block
    """

    # 全局特征提取初始化部分
    # inplanes：输入特征图通道数
    # Scaling_parameter：缩放参数，控制模块内部压缩率，影响后续中间瓶颈层通道数，从而调整模块的复杂度和参数量
    # pool：池化模式选择，avg（平均池化）或者att（注意力池化）
    def __init__(self, inplanes, Scaling_parameter, pool):
        super(GEblock, self).__init__()
        assert pool in ['avg', 'att']
        self.inplanes = inplanes
        self.planes = int(self.inplanes/Scaling_parameter)
        self.pool = pool

        # 空间池化
        if 'att' in pool:
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 通道注意力
        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(self.inplanes, self.planes, kernel_size=1), # 将通道数inplanes压缩到planes，减少参数
            nn.LayerNorm([self.planes, 1, 1]), # 层归一化，此时输入已经是全局变量，所以不需要批归一化
            nn.ReLU(inplace=True),  # 激活
            nn.Conv2d(self.planes, self.inplanes, kernel_size=1)  #将通道数恢复
        )

        self.reset_parameters()  # 参数初始化


    # 如果是注意力池化，选择凯明初始化
    def reset_parameters(self):
        if self.pool == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True

        last_zero_init(self.channel_add_conv)


    # 生成基于全局上下文的通道注意力权重
    def spatial_pool(self, x):

        # 获取输入特征图的形状
        batch, channel, height, width = x.size()
        if self.pool == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width) # 将空间维度展平
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)  # 增加一个维度，为后续矩阵乘法准备
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)  # 通过1*1卷积生成初始注意力图
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)  # 展平空间维度
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)  # 用softmax得到归一化的注意力权重
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(3)  # 增加一个维度
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)  # 加权求和：特征 × 注意力权重
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)  # 调整形状，得到全局特征向量
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)  # 平均池化模式：直接对每个通道求平均

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)  # 执行上面操作得到全局特征向量

        # [N, C, 1, 1]
        channel_add_term = self.channel_add_conv(context) #通过通道注意力网络生成调整量
        out = channel_add_term  # 输出即为调整量，调整量就是不同通道需要关注的比重

        return out
