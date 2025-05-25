import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from model import NetworkCIFAR as Network
from collections import namedtuple
from tqdm import tqdm
import wandb
from torch.utils.data import Dataset, Subset



Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
SNN_DARTS = Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('snn_multistep_3x3', 1), ('snn_multistep_3x3', 1), ('snn_multistep_5x5', 3), ('max_pool_3x3', 0), ('snn_multistep_3x3', 4)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('snn_multistep_5x5', 0), ('skip_connect', 0), ('snn_multistep_3x3', 2), ('skip_connect', 3), ('max_pool_3x3', 2), ('snn_multistep_5x5', 2), ('snn_multistep_3x3', 3)], reduce_concat=range(2, 6)) #mIoU: 0.8560 | Loss: 0.1157 | Avg MACs: 280.51 e-6



# 2025-05-20 15:27:02,806 train Top 1: 69.6360 | train Top 5: 97.4320  | Loss: 0.8680 | Total energy: tensor(0.0017, device='cuda:1', grad_fn=<AddBackward0>)
# 2025-05-20 15:28:32,208 Validation Top 1: 70.8000 | Val Top 5: 97.8440  | Loss: 0.8356
# 2025-05-20 15:28:32,312 epoch 4 lr 9.997780e-03
New_dart= Genotype(normal=[('max_pool_3x3', 1), ('conv_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 1), ('sep_conv_5x5', 3), ('alt_sep_conv', 1), ('sep_conv_3x3', 0), ('skip_connect', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 3), ('max_pool_3x3', 0), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], reduce_concat=range(2, 6))


# 2025-05-20 15:06:27,302 train Top 1: 60.3440 | train Top 5: 95.6640  | Loss: 1.1171 | Total energy: tensor(0.0040, device='cuda:0', grad_fn=<AddBackward0>)
# 2025-05-20 15:08:26,485 Validation Top 1: 65.3320 | Val Top 5: 96.9640  | Loss: 0.9775
# 2025-05-20 15:08:26,627 epoch 2 lr 9.999201e-03
New_dart = Genotype(normal=[('alt_sep_conv', 1), ('alt_sep_conv', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 3), ('max_pool_3x3', 2), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], reduce_concat=range(2, 6))

import torch
import torch.nn as nn


class WeightedFocalDiceLoss(nn.Module):
    def __init__(self, class_frequencies, gamma=2.0, alpha=0.25, bce_weight=0.5, dice_weight=0.5, min_weight=0.1, max_weight=50.0, rare_class_factor=10.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        # Compute weights based on class frequencies
        freq_tensor = torch.tensor(class_frequencies, dtype=torch.float32)
        inverse_freq = 1.0 / (freq_tensor + 1e-6)
        weights = inverse_freq.clone()
        weights[:-1] = weights[:-1] * rare_class_factor  # Tăng hệ số cho các lớp không phải nền
        weights = torch.clamp(weights, min_weight / freq_tensor.max(), max_weight / freq_tensor.min())
        weights = weights / weights.sum() * len(weights)  # Chuẩn hóa
        self.weights = weights
        if torch.cuda.is_available():
            self.weights = self.weights.cuda()

    def forward(self, logits, targets):
        # Apply sigmoid to logits to get probabilities
        logits = logits / (torch.abs(logits).max() + 1e-6)  # Normalize logits
        probs = torch.sigmoid(logits)
        targets = targets.float()

        # Ensure shapes are compatible
        if targets.dim() == 4 and targets.shape[1] == 1:  # Single-label case, convert to one-hot
            num_classes = logits.shape[1]
            targets_one_hot = torch.zeros_like(logits).scatter_(1, targets.long(), 1.0)
            targets = targets_one_hot

        # Compute Focal Loss
        bce = - (targets * torch.log(probs + 1e-6) + (1 - targets) * torch.log(1 - probs + 1e-6))
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_term = (1 - pt) ** self.gamma
        focal_loss = focal_term * bce
        weights_expanded = self.weights.view(1, -1, 1, 1).expand_as(focal_loss)
        focal_loss_weighted = focal_loss * weights_expanded
        focal_loss_final = focal_loss_weighted.mean()

        # Compute Dice Loss
        intersection = (probs * targets).sum(dim=(0, 2, 3))
        union = probs.sum(dim=(0, 2, 3)) + targets.sum(dim=(0, 2, 3))
        dice_loss = 1 - (2.0 * intersection + 1e-6) / (union + 1e-6)
        dice_loss = (dice_loss * self.weights).mean()

        # Combine losses
        total_loss = self.bce_weight * focal_loss_final + self.dice_weight * dice_loss
        return total_loss

import torch
import torch.nn as nn

class WeightedBCELoss(nn.Module):
    def __init__(self, weights=None, reduction='mean'):
        super().__init__()
        self.weights = weights
        self.reduction = reduction
        if weights is not None:
            self.weights = torch.tensor(weights, dtype=torch.float32)
    def forward(self, logits, targets):
        # Áp dụng sigmoid để chuyển logits thành xác suất
        self.weights=self.weights.to(logits.device)
        probs = torch.sigmoid(logits)
        targets = targets.float()

        # Đảm bảo kích thước đầu vào phù hợp
        if targets.dim() == 1:  # Nếu nhãn là 1D (một mẫu), mở rộng thành [1, C]
            targets = targets.unsqueeze(0)
        if logits.dim() == 1:  # Nếu logits là 1D (một mẫu), mở rộng thành [1, C]
            logits = logits.unsqueeze(0)
        if targets.size(1) != logits.size(1):
            raise ValueError(f"Số lớp trong targets ({targets.size(1)}) phải khớp với logits ({logits.size(1)})")

        # Tính BCE cho từng lớp
        bce = - (targets * torch.log(probs + 1e-6) + (1 - targets) * torch.log(1 - probs + 1e-6))

        # Áp dụng trọng số nếu có
        if self.weights is not None:
            if len(self.weights) != bce.size(1):
                raise ValueError(f"Độ dài của weights ({len(self.weights)}) phải khớp với số lớp ({bce.size(1)})")
            weights_expanded = self.weights.view(1, -1).expand_as(bce)
            bce_weighted = bce * weights_expanded
            if self.reduction == 'mean':
                loss = bce_weighted.mean()
            elif self.reduction == 'sum':
                loss = bce_weighted.sum()
            else:
                loss = bce_weighted
        else:
            if self.reduction == 'mean':
                loss = bce.mean()
            elif self.reduction == 'sum':
                loss = bce.sum()
            else:
                loss = bce

        return loss

def analyze_labels(dataset, num_classes=8):
    """Analyze label frequency in the dataset for multilabel classification."""
    class_counts = np.zeros(num_classes)
    total_samples = 0

    # Duyệt qua tập dữ liệu để đếm số lần xuất hiện của từng lớp
    for _, label in tqdm(dataset, desc="Analyzing labels"):
        # label là tensor 1D có shape [num_classes] với giá trị 0 hoặc 1
        class_counts += label.cpu().numpy()  # Cộng dồn số lần xuất hiện của từng lớp
        total_samples += 1

    # Tính tần suất của từng lớp
    class_frequencies = class_counts / total_samples
    logging.info("Class frequencies: %s", class_frequencies)
    logging.info("Total samples analyzed: %d", total_samples)

    # Tính trọng số dựa trên nghịch đảo tần suất
    weights = 1.0 / (class_frequencies + 1e-6)  # Tránh chia cho 0
    weights = weights / weights.sum() * num_classes  # Chuẩn hóa để tổng trọng số bằng num_classes
    logging.info("Computed weights: %s", weights)

    return class_frequencies.tolist(), weights.tolist()
    
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2 * intersection + self.eps) / (union + self.eps)
        return 1 - dice.mean()

class CustomPTSegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.images_dir = os.path.join(root_dir, "X")
        self.labels_dir = os.path.join(root_dir, "vector_label")
        self.transform = transform

        self.image_paths = sorted(glob.glob(os.path.join(self.images_dir, '*.pt')))
        self.label_paths = [os.path.join(self.labels_dir, os.path.basename(p)) for p in self.image_paths]
        assert all([os.path.exists(p) for p in self.label_paths]), "Missing label files."

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = torch.load(self.image_paths[idx]).float()  # Normalize
        label = torch.load(self.label_paths[idx]).float()
        # image = image.permute(0, 2, 1)  # [C, H, W]
        # label = label.permute(0, 2, 1)
        label = label[:8]
        if self.transform:
            image, label = self.transform(image, label)

        # print(f"size anh : {image.size()}, {label.size()}")
        return image, label

parser = argparse.ArgumentParser("segmentation")
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--arch', type=str, default='DARTS')
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.01,help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--snn_step', type=int, default=5, help='Number of time-steps for SNN operations')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--device', type=str, default='cuda', help='device để train: cpu hoặc cuda')
parser.add_argument('--epochs', type=int, default=3, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=2, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=0.01, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')

args = parser.parse_args()




args.save = './log_train/'  
os.makedirs(args.save, exist_ok=True)
timestamp = time.strftime("%Y%m%d-%H%M%S")
log_file_path = os.path.join(args.save, f"log_{timestamp}.txt")

log_format = '%(asctime)s %(message)s'
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format=log_format,
    datefmt='%m/%d %I:%M:%S %p'
)
fh = logging.FileHandler(log_file_path)  # lưu log ra file log.txt trong thư mục hiện tại
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2 * intersection + self.eps) / (union + self.eps)
        return 1 - dice.mean()
class WeightedFocalLoss(nn.Module):
    def __init__(self, class_frequencies, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.weights = torch.tensor(class_frequencies).reciprocal()  # Keep on CPU
    
    def forward(self, logits, targets):
        weights = self.weights.to(logits.device)  # Move to GPU during forward
        bce_loss = nn.BCEWithLogitsLoss(weight=weights)(logits, targets)
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_loss = -self.alpha * (1 - p_t) ** self.gamma * torch.log(p_t + 1e-6)
        return focal_loss.mean()

def compute_metrics(preds, labels, threshold=0.5, double_indices=[0,2,4,5,6]):
    preds = torch.sigmoid(preds)
    if double_indices is not None:
        # double_indices: danh sách hoặc tensor các chỉ số cần nhân
        mask = torch.zeros_like(preds)
        mask[:, double_indices] = 1
        preds = preds * (1 + mask)
    preds = preds > threshold
    labels = labels > 0.5
    tp = (preds & labels).sum(dim=0).float()
    fp = (preds & ~labels).sum(dim=0).float()
    fn = (~preds & labels).sum(dim=0).float()
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    accuracy = (preds == labels).float().mean(dim=0)
    return {
        'accuracy': accuracy.mean().item(),
        'f1_macro': f1.mean().item(),
        'precision': 0,
        'recall': 0
    }
def main():
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True
    cudnn.enabled = True

    genotype = New_dart
    model = Network(36, 10,args.layers, False, genotype,3,1,device).to(device)


    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

    num_train = len(train_data)
    # num_train=500
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True, num_workers=2)
    
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    for epoch in range(args.epochs):
        current_lr = scheduler.get_last_lr()[0]
        logging.info("Epoch %d: Learning Rate %.6f", epoch, current_lr)

        train_top1, train_top5, train_objs =train(train_queue, model,  criterion, optimizer, device ,args)
        logging.info("train Top 1: %.4f | train Top 5: %.4f  | Loss: %.4f | Total energy: %s", train_top1, train_top5, train_objs)

        valid_top1, valid_top5, valid_objs, =evaluate(valid_queue, model, criterion, device)
        
        logging.info("Validation Top 1: %.4f | Val Top 5: %.4f  | Loss: %.4f",valid_top1,  valid_top5, valid_objs)
        wandb.log({
            "epoch/train_Top1": train_top1,
            "epoch/train_loss": train_objs,
            "epoch/train_Top5":train_top5,
            "epoch/val_Top1": valid_top1,
            "epoch/val_loss": valid_objs,
            "epoch/valid_top5":valid_top5,
            "epoch": epoch
        })

        
        torch.save(model.state_dict(), os.path.join(args.save, f"model_{epoch}.pt"))
        scheduler.step()

def train(loader, model, criterion, optimizer, device, agrs):
    epoch=args.epochs
    model.train()
    total_loss = 0
    total_metrics = {'accuracy': 0, 'f1_macro': 0, 'precision': 0, 'recall': 0}
    total_samples = 0
    objs = utils.AvgrageMeter()
    # metrics_avg = {'accuracy': utils.AvgrageMeter(), 'f1_macro': utils.AvgrageMeter()}
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()


    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"[Train Epoch {epoch}]")

    for num_batch, (input_train, target_train) in pbar:
        n = input_train.size(0)

        input_train = input_train.to(device)
        target_train = target_train.to(device)
        
        # optimizer.zero_grad()
        # logits, _ = model(x)
        # loss = criterion(logits, y)
        # loss.backward()
        # optimizer.step()
        
        optimizer.zero_grad()
        logits, energy = model(input_train)
        loss_task = criterion(logits, target_train)
        loss_task.backward()
        optimizer.step()
        prec1, prec5 = utils.accuracy(logits, target_train, topk=(1, 5))
        objs.update(loss_task.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
        wandb.log({
            "train/step_loss": objs.avg,
            "train/top_1": top1.avg,
            "train/top_5": top5.avg,
            "train/step_energy": energy,
        })
        pbar.set_postfix({
            "loss_task": objs.avg,
            "accuracy": top5.avg,
            # "f1_macro": metrics_avg['f1_macro'].avg,
            "energy": f"{energy}"
        })

    
    return top1.avg, top5.avg, objs.avg

def evaluate(loader, model, criterion, device):
    model.eval()
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    pbar = tqdm(loader, desc="[Valid]")

    with torch.no_grad():
        for input_valid, target_valid in pbar:
            input_valid = input_valid.to(device)
            target_valid = target_valid.to(device)
            logits,_ = model(input_valid)
            loss = criterion(logits, target_valid)
            prec1, prec5 = utils.accuracy(logits, target_valid, topk=(1, 5))
            n = input_valid.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            pbar.set_postfix({
                "val_loss": objs.avg,
                "val_accuracy": top1.avg,
            })

            
    return top1.avg, top5.avg, objs.avg

if __name__ == '__main__':
    wandb.init(project="segmentation-darts", name=args.save)
    wandb.config.update(args)
    main()
