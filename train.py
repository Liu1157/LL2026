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
        print("Checkpoint keys:", checkpoint.keys())
        source_state = checkpoint
        print(f'loaded model {pretrained_path}')

        target_state = net.state_dict()
        new_target_state = collections.OrderedDict()
        for target_key, target_value in target_state.items():
            if target_key in source_state and source_state[target_key].size() == target_state[target_key].size():
                new_target_state[target_key] = source_state[target_key]
            else:
                new_target_state[target_key] = target_state[target_key]
                # print('[WARNING] Not found pre-trained parameters for {}'.format(target_key))

        net.load_state_dict(new_target_state)
        print("done ...")
    else:
        print('No pretrained weights provided. Training from scratch.')
        # 不需要 source_state

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

    optimizer = torch.optim.SGD(net.parameters(), lr=parser.lr, momentum=0.99, weight_decay=0.00001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2)

    avg_loss = 0
    losses = []
    print('training ...')

    for epoch in range(parser.epoch):
        epoch_start_time = time.time()
        train_loss = Average()
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

            optimizer.zero_grad()
            outputs = net(img_train)

            if parser.step2:
                loss1 = criterion1(outputs, label_one_hot)
                loss2 = criterion4(outputs, label_one_hot)
                loss = loss1 * parser.w1_bce + loss2 * parser.w2_per
            else:
                loss = criterion1(outputs, label_one_hot)
                loss = (loss * mask).sum() / mask.sum()

            loss.backward()
            optimizer.step()
            train_loss.update(loss.item())

            if index % 1000 == 0:
                losses.append(loss)
                if parser.step2:
                    log_message = "Epoch: [%2d], step: [%2d], loss: [%.8f], BCE_loss: [%.8f], perceptual_loss: [%.8f]" % (
                        (epoch + 1), index, loss, loss1, loss2)
                else:
                    log_message = "Epoch: [%2d], step: [%2d], loss: [%.8f]" % ((epoch + 1), index, loss.detach().item())
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

        avg_loss = train_loss.avg  # 当前epoch的平均训练损失

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

                output = net(img_valid)
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
        epoch_elapsed_time = time.time() - epoch_start_time
        epoch_summary = f"Epoch {epoch + 1}/{parser.epoch}, "
        epoch_summary += f"Train Loss: {avg_loss:.8f}, "
        epoch_summary += f"Validation Loss: {current_val_loss:.8f}, "
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