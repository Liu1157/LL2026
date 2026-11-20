from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torchvision.transforms.functional as tf
from PIL import Image
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import random


def add_salt_pepper_noise(X_imgs):
    # Need to produce a copy as to not modify the original image
    X_imgs_copy = X_imgs.copy()
    row, col, _ = X_imgs_copy[0].shape
    salt_vs_pepper = 0.2
    amount = 0.004
    num_salt = np.ceil(amount * X_imgs_copy[0].size * salt_vs_pepper)
    num_pepper = np.ceil(amount * X_imgs_copy[0].size * (1.0 - salt_vs_pepper))

    for X_img in X_imgs_copy:
        # Add Salt noise
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in X_img.shape]
        X_img[coords[0], coords[1], :] = 1

        # Add Pepper noise
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in X_img.shape]
        X_img[coords[0], coords[1], :] = 0
    return X_imgs_copy


class TrainDataset(Dataset):
    def __init__(self, csv_path):
        """
        Args:
            csv_path (string): csv file path
        """
        # read csv file
        self.data_info = pd.read_csv(csv_path, header=None)
        # The first column of the file contains the first image
        self.image1_arr = np.asarray(self.data_info.iloc[:, 0])
        # The second column of the file contains the second image
        self.image2_arr = np.asarray(self.data_info.iloc[:, 1])
        # The third column of the file contains the label
        self.label_arr = np.asarray(self.data_info.iloc[:, 2])
        # compute length
        self.data_len = len(self.data_info.index)

    def __len__(self):
        return self.data_len

    def add_gaussian_noise(self, image_in, noise_sigma):
        image_in = np.array(image_in)
        temp_image = np.float64(np.copy(image_in))

        h = temp_image.shape[0]
        w = temp_image.shape[1]
        noise_sigma = noise_sigma * 0.03  # ✅ 关键修改
        noise = np.random.randn(h, w) * noise_sigma

        noisy_image = np.zeros(temp_image.shape, np.float64)
        if len(temp_image.shape) == 2:
            noisy_image = temp_image + noise
        else:
            noisy_image[:, :, 0] = temp_image[:, :, 0] + noise
            noisy_image[:, :, 1] = temp_image[:, :, 1] + noise
            noisy_image[:, :, 2] = temp_image[:, :, 2] + noise
        """
        print('min,max = ', np.min(noisy_image), np.max(noisy_image))
        print('type = ', type(noisy_image[0][0][0]))
        """
        noisy_image = Image.fromarray(np.uint8(noisy_image))
        return noisy_image

    def transform(self, img1, img2, label):
        angle = transforms.RandomRotation.get_params([-180, 180])
        # rotate
        img1 = tf.rotate(img1, angle, interpolation=Image.NEAREST)
        img2 = tf.rotate(img2, angle, interpolation=Image.NEAREST)
        label = tf.rotate(label, angle, interpolation=Image.NEAREST)
        # flip
        if random.random() > 0.5:  # 50%概率进行翻转
            flip_type = random.choice(['horizontal', 'vertical'])  # 随机选择翻转类型
            if flip_type == 'horizontal':
                img1 = tf.hflip(img1)
                img2 = tf.hflip(img2)
                label = tf.hflip(label)
            else:  # vertical
                img1 = tf.vflip(img1)
                img2 = tf.vflip(img2)
                label = tf.vflip(label)
        # resize
        H, W = 256, 256
        if random.random() > 0.5:
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                img1, scale=(0.5, 1.5), ratio=(1, 1))
            img1 = tf.resized_crop(img1, i, j, h, w, (H, W))
            img2 = tf.resized_crop(img2, i, j, h, w, (H, W))
            label = tf.resized_crop(label, i, j, h, w, (H, W))
        else:
            pad = random.randint(0, 192)
            img1 = tf.pad(img1, pad)
            img1 = tf.resize(img1, (H, W))
            img2 = tf.pad(img2, pad)
            img2 = tf.resize(img2, (H, W))
            label = tf.pad(label, pad)
            label = tf.resize(label, (H, W))
        # add gaussian noise
        parm_noise = np.random.uniform(0.0, 0.5)
        img1 = self.add_gaussian_noise(img1, parm_noise)
        img2 = self.add_gaussian_noise(img2, parm_noise)
        # convert to tensor and normalize
        img1 = tf.to_tensor(img1)
        img1 = tf.normalize(img1, [0.5], [0.5])
        img2 = tf.to_tensor(img2)
        img2 = tf.normalize(img2, [0.5], [0.5])
        label = tf.to_tensor(label)
        return img1, img2, label

    def __getitem__(self, index):
        # get file name
        image1_name = self.image1_arr[index]
        image2_name = self.image2_arr[index]
        image_label = self.label_arr[index]

        # 从完整路径中提取基础文件名（不带路径和扩展名）
        # 使用image1的文件名作为标识
        base_filename = os.path.splitext(os.path.basename(image1_name))[0]

        # read image file
        img_as_img1 = Image.open(image1_name).convert('L')
        img_as_img2 = Image.open(image2_name).convert('L')
        # read image label
        img_label = Image.open(image_label).convert('L')
        # transform
        img1, img2, img_label = self.transform(img_as_img1, img_as_img2, img_label)


        # ------------------- 新增：GT和Mask处理（核心修复）-------------------
        # 1. 生成Mask：填充区（enhance.py里设的128）为0，有效区为1
        # ToTensor后128会映射为128/255≈0.502，用这个值精准判断
        mask = torch.ones_like(img_label)
        mask[img_label == 128 / 255.0] = 0.0

        # 2. 二值化GT：255（清晰）→1，0（模糊）→0，128（填充）→0
        label_binary = torch.zeros_like(img_label)
        label_binary[img_label > 0.5] = 1.0  # 大于0.5的都是清晰区域


        # # test
        # self.imshow(img_train[0].squeeze(),title='img1')
        # self.imshow(img_train[1].squeeze(),title='img2')
        # self.imshow(img_label.squeeze(),title='label')
        # print('img_train[0]',img_train[0].shape)
        # print('img_train[1]',img_train[1].shape)
        # print('img_label',img_label.shape)
        # np.savetxt('img1.csv', img_train[0].squeeze(), delimiter=',')
        # np.savetxt('img2.csv', img_train[1].squeeze(), delimiter=',')
        # np.savetxt('label.csv', img_label.squeeze(), delimiter=',')
        # img1 = pd.DataFrame(list(img_train[0].squeeze()))
        # img1.to_csv('img1.txt')
        # img2 = pd.DataFrame(list(img_train[1].squeeze()))
        # img2.to_csv('img2.txt')
        # label = pd.DataFrame(list(img_label.squeeze()))
        # label.to_csv('label.txt')
        img_train = torch.cat((img1, img2), dim=0)

        return img_train, label_binary, mask, base_filename

    def imshow(self, tensor, title=None):
        image = tensor.clone()  # we clone the tensor to not do changes on it
        image = image.squeeze(0)  # remove the fake batch dimension
        unloader = transforms.ToPILImage()
        image = unloader(image)
        plt.imshow(image)
        if title is not None:
            plt.title(title)
        plt.pause(2)  # pause a bit so that plots are updated


class Valid_Dataset(Dataset):
    def __init__(self, transforms=None, path_val=''):
        # 读取验证集图像路径
        img1_path = os.path.join(path_val, 'image1')
        img2_path = os.path.join(path_val, 'image2')
        root_path1 = os.path.abspath(img1_path)
        root_path2 = os.path.abspath(img2_path)

        self.img1_list = [os.path.join(root_path1, x) for x in os.listdir(img1_path) if x.endswith('.png')]
        self.img2_list = [os.path.join(root_path2, x) for x in os.listdir(img2_path) if x.endswith('.png')]

        # 【关键修复】获取正确的mask文件夹路径
        # 从validation目录向上退一级，然后找mask文件夹
        validation_dir = os.path.abspath(path_val)  # 获取validation目录的绝对路径
        project_root = os.path.dirname(validation_dir)  # 获取项目根目录（validation的父目录）
        mask_dir = os.path.join(project_root, 'mask')  # 构建mask文件夹路径

        # 验证mask文件夹是否存在
        if not os.path.exists(mask_dir):
            raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

        # 构建标签路径列表
        self.label_list = []
        for img1_path in self.img1_list:
            # 获取文件名
            filename = os.path.basename(img1_path)
            # 构建mask文件夹中的完整路径
            label_path = os.path.join(mask_dir, filename)

            # 验证标签文件是否存在
            if not os.path.exists(label_path):
                # 如果找不到完全相同的文件名，可以尝试其他匹配方式
                # 例如，检查是否有类似命名的文件
                print(f"Warning: Label file not found: {label_path}")
                # 可以在这里添加逻辑处理文件名不匹配的情况
                # 比如，去掉或修改前缀/后缀

            self.label_list.append(label_path)

        # 排序确保对应关系正确
        self.img1_list.sort()
        self.img2_list.sort()
        self.label_list.sort()

        self.transforms = transforms
        print(f"[Valid_Dataset] Found {len(self.img1_list)} validation samples and {len(self.label_list)} labels.")
        print(f"Mask directory: {mask_dir}")

    def __getitem__(self, index):
        # 读取两幅源图像
        image1 = Image.open(self.img1_list[index]).convert('L')
        image2 = Image.open(self.img2_list[index]).convert('L')

        # 读取对应的标签（完全保留你确认的mask文件夹路径逻辑）
        label_path = self.label_list[index]
        if not os.path.exists(label_path):
            print(f"Creating dummy label for missing file: {label_path}")
            dummy_label = Image.new('L', image1.size, 0)
            label = dummy_label
        else:
            label = Image.open(label_path).convert('L')

        # 统一调整尺寸
        target_size = (256, 256)
        image1 = image1.resize(target_size, Image.BILINEAR)
        image2 = image2.resize(target_size, Image.BILINEAR)
        label = label.resize(target_size, Image.NEAREST)

        if self.transforms is not None:
            image1 = self.transforms(image1)
            image2 = self.transforms(image2)
        else:
            image1 = transforms.ToTensor()(image1)
            image2 = transforms.ToTensor()(image2)

        # 标签转为张量
        label = transforms.ToTensor()(label)

        # 防御性检查：验证集GT不应包含128
        if torch.any((label > 0.1) & (label < 0.9)):
            print(f"⚠️ Warning: Validation label {label_path} contains non-binary values!")

        # ------------------- 新增：验证集Mask和GT二值化（对齐训练逻辑）-------------------
        # 验证集GT来自mask文件夹，无128填充，所以Mask全为1
        mask = torch.ones_like(label)
        # 二值化GT：255→1，0→0（和训练集逻辑完全一致）
        label_binary = torch.zeros_like(label)
        label_binary[label > 0.5] = 1.0
        # ------------------------------------------------------------------------

        img_val = torch.cat((image2, image1), dim=0)
        base_filename = os.path.splitext(os.path.basename(self.img1_list[index]))[0]

        # 返回4个值，和训练集完全对齐，避免train.py读取时报错
        return img_val, label_binary, mask, base_filename

    def __len__(self):
        return len(self.img1_list)


class Test_Dataset(Dataset):
    def __init__(self, transforms=None, path_test=''):
        img1_path = os.path.join(path_test, '1')  # ✅ 改为子目录 1
        img2_path = os.path.join(path_test, '2')  # ✅ 改为子目录 2
        root_path1 = os.path.abspath(img1_path)
        root_path2 = os.path.abspath(img2_path)
        self.img1_list = [os.path.join(root_path1, x) for x in os.listdir(img1_path) if
                          x.endswith(('.png', '.jpg', '.jpeg'))]  # ✅ 支持多种格式
        self.img2_list = [os.path.join(root_path2, x) for x in os.listdir(img2_path) if
                          x.endswith(('.png', '.jpg', '.jpeg'))]  # ✅ 支持多种格式
        self.transforms = transforms

    def __getitem__(self, index):
        # stuff
        image1 = Image.open(self.img1_list[index]).convert('L')
        image2 = Image.open(self.img2_list[index]).convert('L')
        if self.transforms is not None:
            image1 = self.transforms(image1)
            image2 = self.transforms(image2)
        img_test = torch.cat((image1, image2), dim=0)
        return img_test

    def __len__(self):
        return len(self.img1_list)
