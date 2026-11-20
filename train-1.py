from utils.utils import parse_args, import_model, PerceptualLoss, setting_cuda
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from data_loader import TrainDataset, Valid_Dataset
from torch import nn
import os
import random
import imageio
import torch
import numpy as np
import collections
import time
import torch.nn.functional as F

parser = parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# Set random number seed
setup_seed(66)
# Preprocess and load data
transformations = transforms.Compose([transforms.Resize((256, 256)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.5], [0.5])])

train_set = TrainDataset(parser.train_csv)
valid_set = Valid_Dataset(transforms=transformations, path_val=parser.val)

print("preparing training data ...")
train_loader = DataLoader(train_set, batch_size=parser.batchsize_train, shuffle=True, drop_last=True, num_workers=5)
print("done ...")
print("preparing valid data ...")
valid_loader = DataLoader(valid_set, batch_size=parser.batchsize_valid, shuffle=False, num_workers=3)
print("done ...")

# load pre-trained _model
os.environ["CUDA_VISIBLE_DEVICES"] = parser.gpus
net = import_model(parser.model, parser.model_inchannel, parser.model_outchannel)

# 加载预训练模型（权重）
if parser.weights is not None:
    pretrained_path = parser.weights
    print("loaded model %s" % pretrained_path)

    if pretrained_path:  # 只有权重路径非空时才加载
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        #print("Checkpoint keys:", checkpoint.keys())
        source_state = checkpoint
        new_source_state = {}
        for key, value in source_state.items():
            new_key = key.replace('module.', '', 1)  # 只替换第一次出现的 'module.'
            new_source_state[new_key] = value
        source_state = new_source_state

        target_state = net.state_dict()


        new_target_state = collections.OrderedDict()
        for target_key, target_value in target_state.items():
            if target_key in source_state and source_state[target_key].size() == target_state[target_key].size():
                new_target_state[target_key] = source_state[target_key]
            else:
                new_target_state[target_key] = target_state[target_key]
                # print('[WARNING] Not found pre-trained parameters for {}'.format(target_key))

        matched = 0
        unmatched = 0
        for target_key, target_value in target_state.items():
            if target_key in source_state and source_state[target_key].size() == target_state[target_key].size():
                matched += 1
            else:
                unmatched += 1
        print(f"[DEBUG] Matched params: {matched}, Unmatched: {unmatched}")

        net.load_state_dict(new_target_state)
        # 在加载权重后，验证 calib_head 初始化
        with torch.no_grad():
            calib_conv = net.calib_head.conv[3]

    else:
        print('No pretrained weights provided. Training from scratch.')
        # 不需要 source_state

# ====== 新增：方案A 阶段1 冻结逻辑 ======
if parser.stage == 1:
    print("[Stage 1] Freezing main network parameters. Only training MAE modules (calib_head/recon_conv)...")
    for name, param in net.named_parameters():
        # 假设你的MAE模块名字包含 'calib' 或 'recon'，请根据 unet_model.py 中的实际命名调整
        if 'calib' not in name and 'recon' not in name:
            param.requires_grad = False


        # ====== 新增：固定主网络 BN 统计量 ======
    def freeze_main_bn(module, module_name=''):
        """递归遍历子模块，将不属于 calib/recon 的 BN 层设为 eval 模式"""
        for child_name, child in module.named_children():
            full_name = f"{module_name}.{child_name}" if module_name else child_name
            if isinstance(child, nn.BatchNorm2d):
                # 如果该 BN 层属于主网络（不包含 calib 或 recon），则固定
                if 'calib' not in full_name and 'recon' not in full_name:
                    child.eval()
            else:
                freeze_main_bn(child, full_name)


    freeze_main_bn(net)
# ========================================

# setting cuda if needed GPU存在否，存在的话把模型移动到GPU上
gpus, net = setting_cuda(parser.gpus, net)
is_cuda = len(gpus) >= 1  # 是否已经把模型搬到GPU上了


class Average(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count


def train():
    # 创建日志目录
    log_dir = './_logs'
    os.makedirs(log_dir, exist_ok=True)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(log_dir, f'training_log_{timestamp}.txt')
    log_file = open(log_file_path, 'w')
    log_file.write("=" * 60 + "\n")
    log_file.write(f"Training Log - Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"Model: {parser.model}, Epochs: {parser.epoch}, LR: {parser.lr}\n")
    log_file.write(f"Train set: {len(train_set)} images, Valid set: {len(valid_set)} images\n")
    log_file.write("=" * 60 + "\n\n")

    # 定义损失函数
    criterion1 = nn.BCEWithLogitsLoss()
    if parser.step2:
        criterion4 = PerceptualLoss(is_cuda)
    if is_cuda:
        criterion1 = criterion1.cuda()

    def gradient_loss(pred, target):
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        return (torch.abs(pred_dx - target_dx).mean() + torch.abs(pred_dy - target_dy).mean())

    # ====== 优化器与学习率适配（含分离优化器） ======
    current_lr = parser.lr
    if parser.stage == 2:
        current_lr = parser.lr * 0.1
        print(f"[Stage 2] Finetuning all parameters with reduced LR: {current_lr}")

    if parser.stage == 1:
        # 阶段1：分离优化器
        calib_params = [p for n, p in net.named_parameters() if 'calib' in n and p.requires_grad]
        recon_params = [p for n, p in net.named_parameters() if 'recon' in n and p.requires_grad]

        optimizer_calib = torch.optim.SGD(calib_params, lr=current_lr, momentum=0.99, weight_decay=0.00001)
        optimizer_recon = torch.optim.SGD(recon_params, lr=current_lr * 0.1, momentum=0.99, weight_decay=0.00001)

        # 学习率调度器只作用于 calib_head
        scheduler = ReduceLROnPlateau(optimizer_calib, 'min', factor=0.1, patience=5)

        print(f"[Stage 1] Separated optimizers: calib LR={current_lr}, recon LR={current_lr * 0.1}")
    else:
        # 阶段2或其他：单一优化器
        optimizer = torch.optim.SGD(net.parameters(), lr=current_lr, momentum=0.99, weight_decay=0.00001)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5)
    avg_loss = 0
    losses = []
    print('training ...')

    for epoch in range(parser.epoch):
        epoch_start_time = time.time()
        train_loss = Average()
        bce_loss_avg = Average()  # 新增
        recon_loss_avg = Average()
        calib_mag_avg = Average()
        grad_loss_avg = Average()
        if epoch != 0:
            scheduler.step(avg_loss)
        net.train()

        def set_norm_eval(m):
            classname = m.__class__.__name__
            if classname.find('InstanceNorm') != -1 or classname.find('BatchNorm2d') != -1:
                m.eval()

        if parser.step2:
            print('freezing Normalization')
            net.apply(set_norm_eval)
            print('done')

        # ========== 训练阶段 (Training Phase) ==========
        for index, (img_train, label, mask, filename) in enumerate(train_loader):
            H, W = label.shape[2:]
            label = label.numpy()
            label = torch.LongTensor(label)
            label_one_hot = torch.zeros(parser.batchsize_train, parser.model_inchannel, H, W).scatter_(1, label, 1)

            if is_cuda:
                img_train = img_train.cuda()
                label_one_hot = label_one_hot.cuda()
                mask = mask.cuda()

            # 在 loss.backward() 之前
            if parser.stage == 1:
                optimizer_calib.zero_grad()
                optimizer_recon.zero_grad()
            else:
                optimizer.zero_grad()
            final_out, d2, recon_final = net(img_train)

            outputs = final_out  # 主任务仍用最终决策图计算BCE Loss

            if parser.step2:
                loss1 = criterion1(outputs, label_one_hot)
                loss2 = criterion4(outputs, label_one_hot)
                loss = loss1 * parser.w1_bce + loss2 * parser.w2_per
                recon_loss_tensor = None
                grad_loss_tensor = None
            else:
                loss = criterion1(outputs, label_one_hot)
                loss = (loss * mask).sum() / mask.sum()
                bce_loss_masked = loss.detach().clone()

                # 寄生式重建Loss（仅训练时启用）
                recon_loss_tensor = None
                grad_loss_tensor = None
                if net.training:
                    I1 = img_train[:, 0:1, :, :]
                    I2 = img_train[:, 1:2, :, :]
                    gt_clear = label_one_hot[:, 1:2, :, :] * I1 + label_one_hot[:, 0:1, :, :] * I2

                    recon_loss_tensor = nn.MSELoss()(recon_final, gt_clear)
                    loss = loss + 0.01 * recon_loss_tensor
                    # ====== 新增：梯度损失 ======
                    grad_loss_tensor = gradient_loss(recon_final, gt_clear)
                    loss = loss + 0.1 * grad_loss_tensor

            with torch.no_grad():
                _m = net.module if hasattr(net, 'module') else net
                if hasattr(_m, 'last_calib_output'):
                    calib_mag = _m.last_calib_output.abs().mean().item()
                else:
                    calib_mag = float('nan')

            calib_mag_avg.update(calib_mag)
            loss.backward()



            # 梯度检查（可选，调试时可保留，正式训练建议注释掉）
            # for name, param in net.named_parameters():
            #     if 'calib' in name or 'recon' in name:
            #         if param.grad is not None:
            #             print(f"{name}: grad norm = {param.grad.norm().item():.6f}")
            #         else:
            #             print(f"{name}: grad is None!")

            # 在 loss.backward() 之后
            if parser.stage == 1:
                optimizer_calib.step()
                optimizer_recon.step()
            else:
                optimizer.step()
            # 获取各分量的标量值
            total_loss_item = loss.item()
            bce_loss_item = bce_loss_masked.item()
            recon_loss_item = recon_loss_tensor.item() if recon_loss_tensor is not None else 0.0
            grad_loss_item = grad_loss_tensor.item() if grad_loss_tensor is not None else 0.0

            # 更新四个累加器
            train_loss.update(total_loss_item)
            bce_loss_avg.update(bce_loss_item)
            recon_loss_avg.update(recon_loss_item)
            grad_loss_avg.update(grad_loss_item)

            if index % 1000 == 0:
                if parser.step2:
                    log_message = "Epoch: [%2d], step: [%2d], loss: [%.8f], BCE_loss: [%.8f], recon_loss: [%.8f], calib_mag: [%.6f]" % (
                        (epoch + 1), index, loss.detach().item(), bce_loss_item, recon_loss_item, calib_mag)
                else:
                    log_message = "Epoch: [%2d], step: [%2d], loss: [%.8f], BCE_loss: [%.8f], recon_loss: [%.8f], grad_loss: [%.8f], calib_mag: [%.6f]" % (
                        (epoch + 1), index, loss.detach().item(), bce_loss_item, recon_loss_item, grad_loss_item,
                        calib_mag)
                print(log_message)
                log_file.write(log_message + "\n")
                log_file.flush()

            # --- 每个epoch结束时，保存训练预测图和模型 ---
            if index == len(train_loader) - 1:
                i = 0
                current_filename = filename[i]

                output = outputs[i, :, :, :].unsqueeze(0)
                output = output.cpu().detach().numpy()
                result = output.astype(np.float32) * 255.
                result = np.clip(result, 0, 255).astype('uint8')

                confidence_map = np.argmax(result, axis=1)
                confidence_map = confidence_map.squeeze()
                confidence_map = (confidence_map * 255).astype(np.uint8)

                label1 = label[i, :, :, :].squeeze()
                label1 = label1 * 255
                label1 = label1.numpy().astype(np.uint8)

                pred_dir = './_paper_figures/training_progress'
                os.makedirs(pred_dir, exist_ok=True)
                epoch_str = f"{epoch + 1:03d}"
                pred_path = os.path.join(pred_dir, f'{current_filename}_epoch_{epoch_str}_pred.png')
                imageio.imwrite(pred_path, confidence_map)

                if epoch == 0:
                    label_dir = './_paper_figures/ground_truth'
                    os.makedirs(label_dir, exist_ok=True)
                    label_path = os.path.join(label_dir, f'{current_filename}_gt.png')
                    imageio.imwrite(label_path, label1)
                    print(f"Saved ground truth image: {label_path}")

                print(f"Epoch {epoch + 1:03d}: Saved prediction -> {pred_path}")

                os.makedirs('_model', exist_ok=True)
                torch.save(net.state_dict(), '_model/epoch_{:03d}.pth'.format(epoch + 1))

        avg_loss = train_loss.avg
        avg_bce = bce_loss_avg.avg
        avg_recon = recon_loss_avg.avg
        epoch_elapsed_time = time.time() - epoch_start_time

        # 打印 epoch 平均 loss（包含三个分量）
        print(
            f"Epoch {epoch + 1}/{parser.epoch}, Train Loss: {avg_loss:.8f}, BCE Loss: {avg_bce:.8f}, Recon Loss: {avg_recon:.8f}")# 当前epoch的平均训练损失
        # ========== 验证阶段 (Validation Phase) ==========
        # 注意：此部分已从“最后一个batch”的判断中移出，现在在每个epoch的所有训练批次完成后执行。
        current_val_loss = 0.0
        with torch.no_grad():
            net.eval()
            valid_loss = Average()
            valid_samples_saved = 0

            for val_batch_idx, (img_valid, valid_label, valid_mask, filename) in enumerate(valid_loader):
                if is_cuda:
                    img_valid = img_valid.cuda()

                current_batch_size = img_valid.size(0)
                if is_cuda:
                    H, W = valid_label.shape[2:]
                    valid_label = valid_label.numpy()
                    valid_label = torch.LongTensor(valid_label)
                    valid_label_one_hot = torch.zeros(
                        current_batch_size,
                        parser.model_inchannel,
                        H, W
                    ).scatter_(1, valid_label, 1)
                    if is_cuda:
                        valid_label_one_hot = valid_label_one_hot.cuda()
                        valid_mask = valid_mask.cuda()

                output, _, _ = net(img_valid)
                loss_per_pixel = criterion1(output, valid_label_one_hot)
                loss_valid = (loss_per_pixel * valid_mask).sum() / valid_mask.sum()
                valid_loss.update(loss_valid.item())

                if valid_samples_saved < 3:
                    for j in range(min(3, current_batch_size)):
                        if valid_samples_saved >= 3:
                            break
                        current_filename = filename[j]
                        output_single = output[j:j + 1, :, :, :]
                        output_cpu = output_single.cpu().detach().numpy()
                        result = output_cpu.astype(np.float32) * 255.
                        result = np.clip(result, 0, 255).astype('uint8')
                        confidence_map = np.argmax(result, axis=1)
                        confidence_map = confidence_map.squeeze()
                        confidence_map = (confidence_map * 255).astype(np.uint8)

                        img_path = os.path.join('./_image/valid_pred',
                                                f"epoch_{epoch + 1:03d}_sample_{current_filename}_pred.png")
                        os.makedirs('./_image/valid_pred', exist_ok=True)
                        imageio.imwrite(img_path, confidence_map)
                        valid_samples_saved += 1

            current_val_loss = valid_loss.avg
            val_log = f"Epoch {epoch + 1:03d}: Validation Loss: {current_val_loss:.8f}"
            print(val_log)
            log_file.write(val_log + "\n")
            log_file.flush()

        # 恢复训练模式
        net.train()
        if parser.step2:
            net.apply(set_norm_eval)
        # ========== 验证阶段结束 ==========

        # 打印并记录每个epoch的完整总结
        epoch_summary = f"Epoch {epoch + 1}/{parser.epoch}, "
        epoch_summary += f"Train Loss: {avg_loss:.8f}, "
        epoch_summary += f"Train BCE: {avg_bce:.8f}, "
        epoch_summary += f"Train Recon: {avg_recon:.8f}, "
        epoch_summary += f"Grad Loss: {grad_loss_avg.avg:.8f}, "
        epoch_summary += f"Validation Loss: {current_val_loss:.8f}, "
        epoch_summary += f"Calib_Mag: {calib_mag_avg.avg:.6f}, "
        epoch_summary += f"Time: {epoch_elapsed_time:.2f}s"
        print(epoch_summary)
        log_file.write(epoch_summary + "\n")
        log_file.flush()

    # 训练结束后
    log_file.write("\n" + "=" * 60 + "\n")
    log_file.write(f"Training Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write("=" * 60 + "\n")
    log_file.close()
    print(f"\n训练日志已保存到: {log_file_path}")


if __name__ == "__main__":
    train()