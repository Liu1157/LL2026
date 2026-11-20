from data_loader import Test_Dataset
from torchvision import transforms
from utils.utils import parse_args, import_model, setting_cuda
from torch.utils.data import DataLoader
import torch
import os
import collections
import numpy as np
import scipy
import fusion_morphology
from PIL import Image

version = "v3"
type = "maskEdge"

parser = parse_args()
# Preprocess and load data
transformations = transforms.Compose([
    transforms.Resize((256, 256)),  # ✅ 必须加！训练和推理输入尺寸必须一致
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # ✅ 必须改！和训练的[-1,1]归一化对齐
])
test_set = Test_Dataset(transforms=transformations, path_test=parser.test)

test_loader = DataLoader(test_set, batch_size=parser.batchsize_test, shuffle=False)

# load pre-trained _model
os.environ["CUDA_VISIBLE_DEVICES"] = parser.gpus

net = import_model(parser.model, parser.model_inchannel, parser.model_outchannel)

try:
    pretrained_path = parser.weights
    print("loaded model %s" % pretrained_path)

    checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
    print(f"加载检查点 '{pretrained_path}'， 共 {len(checkpoint)} 个键")

    # 1. 去除 `module.` 前缀
    new_checkpoint = {}
    for k, v in checkpoint.items():
        if k.startswith('module.'):
            new_key = k[7:]  # 去除 'module.'
        else:
            new_key = k
        new_checkpoint[new_key] = v

    source_state = new_checkpoint
    target_state = net.state_dict()

    # 2. 详细对比和强制加载
    print("\n--- 开始权重匹配 ---")
    matched_keys = []
    unmatched_target_keys = []
    unmatched_source_keys = list(source_state.keys())  # 初始化为所有源键

    for target_key, target_param in target_state.items():
        if target_key in source_state and source_state[target_key].shape == target_param.shape:
            matched_keys.append(target_key)
            if target_key in unmatched_source_keys:
                unmatched_source_keys.remove(target_key)
        else:
            unmatched_target_keys.append(target_key)

    print(f"匹配成功: {len(matched_keys)} 个键")
    print(f"模型中未匹配: {len(unmatched_target_keys)} 个键")
    print(f"权重中未使用: {len(unmatched_source_keys)} 个键")

    if len(unmatched_target_keys) > 0 and len(matched_keys) > 0:
        print("\n[警告] 部分模型参数未从权重文件加载。尝试加载匹配的参数...")
        # 构建新的状态字典，优先使用匹配的权重，缺失的保留模型随机初始化
        new_target_state = target_state.copy()  # 先复制模型当前状态（随机初始化）
        for key in matched_keys:
            new_target_state[key] = source_state[key]  # 用权重文件中的值覆盖
        net.load_state_dict(new_target_state)
        print("已加载匹配的参数，未匹配的参数保持随机初始化。")
    elif len(matched_keys) == 0:
        print("\n[错误] 没有任何参数匹配！模型将完全随机初始化。")
    else:
        # 理想情况：完全匹配
        net.load_state_dict(source_state)
        print("所有权重匹配成功，已加载。")

    target_state = net.state_dict()
    new_target_state = collections.OrderedDict()
    for target_key, target_value in target_state.items():
        if target_key in source_state and source_state[target_key].size() == target_state[
            target_key].size():
            new_target_state[target_key] = source_state[target_key]
        else:
            new_target_state[target_key] = target_state[target_key]
            print('[WARNING] Not found pre-trained parameters for {}'.format(target_key))
    net.load_state_dict(new_target_state)
    print("done ...")
except Exception as e:
    print('load model error')
    print(e)

# setting cuda if needed
gpus, net = setting_cuda(parser.gpus, net)

is_cuda = len(gpus) >= 1

net.eval()

# 创建保存决策图的目录
save_dir = './image/test_pred/v3/maskEdge/initial_decisionmap/'
os.makedirs(save_dir, exist_ok=True)


for i, img_test in enumerate(test_loader):
    if is_cuda:
        img_test = img_test.cuda()
    output = net(img_test)
    output = output.cpu().detach().numpy()

    # 调试：查看输出形状
    print(f"[DEBUG] Model output shape: {output.shape}")

    # 获取决策图（取通道维度上的最大值索引）
    confidence_map = 1- np.argmax(output, axis=1)

    # 调试：查看决策图形状
    print(f"[DEBUG] Confidence map shape before squeeze: {confidence_map.shape}")

    # 处理批次中的每个样本
    batch_size = confidence_map.shape[0]

    for j in range(batch_size):
        # 获取单个样本的决策图
        single_confidence_map = confidence_map[j]

        # 去除多余的维度
        single_confidence_map = single_confidence_map.squeeze()

        # 调试：查看单个决策图形状
        print(f"[DEBUG] Sample {j} shape after squeeze: {single_confidence_map.shape}")
        print(f"[DEBUG] Sample {j} dtype: {single_confidence_map.dtype}")

        # 确保是二维数组
        if len(single_confidence_map.shape) > 2:
            single_confidence_map = single_confidence_map.squeeze()

        # 将 0/1 值映射到 0/255
        single_confidence_map = (single_confidence_map * 255).astype(np.uint8)

        # 确保数据类型是 uint8
        single_confidence_map = single_confidence_map.astype(np.uint8)

        # 定义保存路径
        img_path = os.path.join(save_dir, f"{i * batch_size + j + 1:03d}.png")

        # 保存图片
        Image.fromarray(single_confidence_map).save(img_path)
        print(f"Saved decision map to: {img_path}")

fusion_morphology.M_fusion(version, type)
print('Finished testing')