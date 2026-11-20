import numpy as np
from skimage import io, morphology
from utils.CRF import crf
import shutil
from skimage.transform import resize  # 用于把决策图还原到原图尺寸

def fusion(initial_decisionmap_path, final_decisionmap_path, image1_path, image2_path, initial_image_save_path, fusion_image_save_path):
    global initial_decisionmap, final_decisionmap
    initial_decisionmap = io.imread(initial_decisionmap_path)
    initial_decisionmap = initial_decisionmap / 255.0
    initial_decisionmap = np.array(initial_decisionmap)
    initial_decisionmap[initial_decisionmap <= 0.5] = 0
    initial_decisionmap[initial_decisionmap > 0.5] = 1

    # ✅ 新增这行：读取CRF输出的软概率图，归一化到0~1（CRF输出是0~255的uint8，除以255得到0~1的连续概率）
    decisionmap = io.imread(final_decisionmap_path).astype(np.float32) / 255.0

    print(f"CRF决策图统计 - 路径: {final_decisionmap_path}")
    print(f"  最大值: {decisionmap.max():.6f}, 最小值: {decisionmap.min():.6f}")
    print(f"  平均值: {decisionmap.mean():.6f}, 标准差: {decisionmap.std():.6f}")
    print(f"  唯一值数量: {len(np.unique(decisionmap))}")

    # 如果是灰度图，检查是否有中间值（不是纯0或纯1）
    unique_vals = np.unique(decisionmap)
    if len(unique_vals) <= 10:  # 如果唯一值很少，打印出来看看
        print(f"  前20个唯一值: {unique_vals[:20]}")
    else:
        print(f"  唯一值太多，只显示范围: [{unique_vals.min():.6f}, {unique_vals.max():.6f}]")

    image1 = io.imread(image1_path)
    image2 = io.imread(image2_path)

    if len(np.array(image1).shape) == 2:
        if len(np.array(decisionmap).shape) == 3:
            final_decisionmap = decisionmap[:, :, 0]
            fusion_image = image1 * final_decisionmap + image2 * (1 - final_decisionmap)
            # save fused image
            io.imsave(fusion_image_save_path, fusion_image.astype(np.uint8))
        else:
            final_decisionmap = decisionmap
            fusion_image = image1 * final_decisionmap + image2 * (1 - final_decisionmap)
            # save fused image
            io.imsave(fusion_image_save_path, fusion_image.astype(np.uint8))

        initial_fusion_image = image1 * initial_decisionmap + image2 * (1 - initial_decisionmap)
        io.imsave(initial_image_save_path, initial_fusion_image.astype(np.uint8))

    else:
        final_decisionmap = np.ones([image1.shape[0], image1.shape[1], image1.shape[2]])
        print("init: ", initial_decisionmap.shape)
        initial_decisionmap = np.expand_dims(initial_decisionmap, 2)
        print("1: ", initial_decisionmap.shape)
        initial_decisionmap = np.repeat(initial_decisionmap, 3, axis=2)
        print("2: ", initial_decisionmap.shape)
        if len(np.array(decisionmap).shape) == 2:
            final_decisionmap[:, :, 0] = decisionmap
            final_decisionmap[:, :, 1] = decisionmap
            final_decisionmap[:, :, 2] = decisionmap
        else:
            final_decisionmap = decisionmap

        fusion_image = image1 * final_decisionmap + image2 * (1 - final_decisionmap)
        # save fused image
        io.imsave(fusion_image_save_path, fusion_image.astype(np.uint8))

        # save initial fused image
        initial_fusion_image = image1 * initial_decisionmap + image2 * (1 - initial_decisionmap)
        io.imsave(initial_image_save_path, initial_fusion_image.astype(np.uint8))


def fusion_main(initial_decisionmap_path, image1_path, image2_path, final_decisionmap_save_path, final_crf_path, initial_image_save_path, fusion_image_save_path):
    import os
    from skimage import io

    # 确保所有输出目录存在
    os.makedirs(os.path.dirname(final_decisionmap_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(final_crf_path), exist_ok=True)
    os.makedirs(os.path.dirname(initial_image_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(fusion_image_save_path), exist_ok=True)

    # ------------------- 核心修复：尺寸对齐 -------------------
    # 1. 读取原图尺寸（比如520x520）
    image1 = io.imread(image1_path)
    h, w = image1.shape[:2]  # 支持灰度图和彩色图
    print(f"DEBUG: 原图尺寸: {h}x{w}, 目标决策图尺寸: {h}x{w}")

    # 2. 读取模型输出的256x256决策图，Resize回原图尺寸
    init_dm = io.imread(initial_decisionmap_path)
    print(f"DEBUG: 模型输出决策图尺寸: {init_dm.shape}")
    # 最近邻插值(order=0)：避免引入非0/255的中间值，保持决策图二值性
    init_dm_resized = resize(init_dm, (h, w), order=0, preserve_range=True).astype(np.uint8)
    print(f"DEBUG: Resize后决策图尺寸: {init_dm_resized.shape}")

    # 3. 保存Resize后的决策图（临时文件，用于CRF和融合）
    resized_dm_path = initial_decisionmap_path.replace('.png', '_resized.png')
    io.imsave(resized_dm_path, init_dm_resized)
    # --------------------------------------------------------

    # 复制Resize后的决策图到CRF输出路径（保留你的调试逻辑）


    # 运行CRF（现在决策图尺寸和原图完全一致，不会再报错）
    crf(image1_path, resized_dm_path, final_crf_path)

    # 运行融合（同样用Resize后的决策图，避免融合时尺寸不匹配）
    fusion(resized_dm_path, final_crf_path, image1_path, image2_path, initial_image_save_path, fusion_image_save_path)

    # 可选：调试完成后删除临时文件（不想删可以注释掉）
    # os.remove(resized_dm_path)
def M_fusion(version, type):
    n = 1
    for i in range(1, 121):
        print(i)
        # ✅ 所有样本都用统一的1/2文件夹路径，和训练集结构完全一致
        initial_decisionmap_path = "./image/test_pred/" + version + "/" + type + "/initial_decisionmap/" + f"{i:03d}" + ".png"
        image1_path = "./data/test/1/" + f"{n:03d}" + ".jpg"  # I1=blur1/1（对应训练第二个输入，GT标注对象）
        image2_path = "./data/test/2/" + f"{n:03d}" + ".jpg"  # I2=blur1/2（对应训练第一个输入）
        final_decisionmap_save_path = "./result/" + version + "/" + type + "/decisionMap_" + str(i) + "_crf.png"
        final_crf_path = "./result/" + version + "/" + type + "/decisionmap_crf/" + str(i) + "_crf.png"
        initial_image_save_path = "./result/" + version + "/" + type + "/initial_fusion/" + str(i) + ".png"
        fusion_image_save_path = "./result/" + version + "/" + type + "/fusion/" + str(i) + "_crf.png"
        fusion_main(initial_decisionmap_path, image1_path, image2_path, final_decisionmap_save_path, final_crf_path,
                    initial_image_save_path, fusion_image_save_path)
        n += 1