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

SNN_DARTS = Genotype(normal=[('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 2), ('dil_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('skip_connect', 1), ('skip_connect', 0), ('dil_conv_3x3', 0), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 3), ('sep_conv_5x5', 2), ('sep_conv_3x3', 3)], reduce_concat=range(2, 6))


New_dart= Genotype(normal=[('conv_1x1', 0), ('conv_1x1', 1), ('dil_conv_3x3', 2), ('grouped_conv', 1), ('max_pool_3x3', 3), ('conv_1x1', 2), ('alt_sep_conv', 3), ('conv_1x1', 4)], normal_concat=range(2, 6), reduce=[('skip_connect', 1), ('conv_1x1', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('grouped_conv', 1), ('sep_conv_5x5', 3), ('grouped_conv', 1), ('sep_conv_3x3', 2)], reduce_concat=range(2, 6))

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
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--grad_clip', type=float, default=5)
parser.add_argument('--save', type=str, default='EXP')
parser.add_argument('--arch', type=str, default='DARTS')
parser.add_argument('--seed', type=int, default=2)
parser.add_argument('--layers', type=int, default=2)
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')

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

# def compute_metrics(preds, labels, threshold=0.5, eps=1e-6):
#     """
#     Compute multi-label metrics: Accuracy, F1 Score (macro), Precision, Recall.
#     preds: Tensor [B, C] - logits (before sigmoid)
#     labels: Tensor [B, C] - binary labels (0/1)
#     """
#     preds = torch.sigmoid(preds) > threshold
#     labels = labels > 0.5
#     preds = preds.float()
#     labels = labels.float()

#     # True Positives, False Positives, False Negatives
#     tp = (preds * labels).sum(dim=0)
#     fp = (preds * (1 - labels)).sum(dim=0)
#     fn = ((1 - preds) * labels).sum(dim=0)

#     # Precision, Recall, F1 per class
#     precision = tp / (tp + fp + eps)
#     recall = tp / (tp + fn + eps)
#     f1 = 2 * precision * recall / (precision + recall + eps)

#     # Macro averages
#     precision_macro = precision.mean().item()
#     recall_macro = recall.mean().item()
#     f1_macro = f1.mean().item()

#     # Accuracy: Average per-class accuracy
#     correct = (preds == labels).float().mean(dim=0)
#     accuracy = correct.mean().item()

#     return {
#         'accuracy': accuracy,
#         'f1_macro': f1_macro,
#         'precision': precision_macro,
#         'recall': recall_macro
#     }


def compute_metrics(preds, labels, threshold=0.5):
    preds = torch.sigmoid(preds) > threshold
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
    model = Network(36, 8,args.layers, False, genotype,4,1,device).to(device)
        
    
    # criterion = nn.BCEWithLogitsLoss().to(device)
    # criterion=DiceLoss().to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # for name, param in model.named_parameters():
    #     print(name, param.shape)

    full_dataset = CustomPTSegmentationDataset(root_dir="../../../Agriculture-Vision-2021_processed_zip/trainrandcrop256/")
    indices = list(range(30000))
    np.random.shuffle(indices)
    train_indices = indices[:25000]
    val_indices = indices[25000:30000]
    trainset = Subset(full_dataset,train_indices)
    class_frequencies,weights = analyze_labels(trainset, 8)
    criterion = WeightedBCELoss(weights=weights, reduction='mean').to(device)
    # criterion=WeightedFocalDiceLoss(class_frequencies)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_loader = torch.utils.data.DataLoader(full_dataset, batch_size=args.batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(train_indices), num_workers=2)
    val_loader = torch.utils.data.DataLoader(full_dataset, batch_size=args.batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(val_indices), num_workers=2)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # for epoch in range(args.epochs):
    #     # if epoch < 20:
    #     #     criterion = utils.BCE_Dice_LovaszLoss(1.0, 1.0, 0.5)
    #     # else:
    #     #     criterion = utils.BCE_Dice_LovaszLoss(1.0, 1.0, 1.5)

    #     # print(f"training epoch {epoch}")
    #     current_lr = scheduler.get_last_lr()[0]
    #     logging.info("Epoch %d: Learning Rate %.6f", epoch, current_lr)
    #     train_loss, train_iou = train(train_loader, model, criterion, optimizer, device, epoch)
        
    #     print(f"total epoch: { epoch }")
    #     val_loss, val_iou = evaluate(val_loader, model, criterion, device)
    #     logging.info("Epoch %d: Train Loss %.4f | Val Loss %.4f | mIoU %.4f", epoch, train_loss, val_loss, val_iou)
    #     wandb.log({
    #             "epoch": epoch,
    #             "train/loss": train_loss,
    #             "train/IoU": train_iou,
    #             "val/loss": val_loss,
    #             "val/IoU": val_iou,
    #             "lr": scheduler.get_last_lr()[0]
    #     })
    #     torch.save(model.state_dict(), os.path.join(args.save, f"model_{epoch}.pt"))
    #     scheduler.step()  # <- thêm dòng này
    for epoch in range(args.epochs):
        current_lr = scheduler.get_last_lr()[0]
        logging.info("Epoch %d: Learning Rate %.6f", epoch, current_lr)
        train_loss, train_metrics = train(train_loader, model, criterion, optimizer, device, epoch)
        val_loss, val_metrics = evaluate(val_loader, model, criterion, device)
        logging.info("Epoch %d: Train Loss %.4f | Val Loss %.4f | Train F1 %.4f | Val F1 %.4f | Train Acc %.4f | Val Acc %.4f",
                     epoch, train_loss, val_loss, train_metrics['f1_macro'], val_metrics['f1_macro'],
                     train_metrics['accuracy'], val_metrics['accuracy'])
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/f1_macro": train_metrics['f1_macro'],
            "train/accuracy": train_metrics['accuracy'],
            "train/precision": train_metrics['precision'],
            "train/recall": train_metrics['recall'],
            "val/loss": val_loss,
            "val/f1_macro": val_metrics['f1_macro'],
            "val/accuracy": val_metrics['accuracy'],
            "val/precision": val_metrics['precision'],
            "val/recall": val_metrics['recall'],
            "lr": current_lr
        })
        torch.save(model.state_dict(), os.path.join(args.save, f"model_{epoch}.pt"))
        scheduler.step()

import wandb

# def train(loader, model, criterion, optimizer, device, epoch):
#     model.train()
#     total_loss = 0
#     total_iou = 0
#     total_samples = 0

#     pbar = tqdm(enumerate(loader), total=len(loader), desc=f"[Train Epoch {epoch}]")

#     for num_batch, (x, y) in pbar:
#         x, y = x.to(device), y.to(device)
#         optimizer.zero_grad()
#         logits, _ = model(x)
#         loss=criterion(logits, y)
#         loss.backward()
#         optimizer.step()

#         batch_size = x.size(0)
#         total_loss += loss.item() * batch_size
#         total_samples += batch_size

#         # Tính miou
#         # dice_score = utils.compute_modified_miou(logits, y)
#         dice_score=0
#         total_iou += dice_score * batch_size
        
    
#         # Trung bình tới thời điểm hiện tại
#         avg_loss = total_loss / total_samples
#         avg_iou = total_iou / total_samples

#         # --- Log wandb tại mỗi step ---
#         global_step = num_batch + epoch * len(loader)
#         wandb.log({
#             "train/avg_loss": loss.item(),
#             "train/avg_miou": dice_score,
#         }, step=global_step)

#         pbar.set_postfix(loss=avg_loss, miou=avg_iou)

#     return avg_loss, avg_iou

# def evaluate(loader, model, criterion, device):
#     model.eval()
#     total_loss = 0
#     total_iou = 0
#     total_samples = 0

#     pbar = tqdm(loader, desc="[Valid]")

#     with torch.no_grad():
#         for x, y in pbar:
#             x, y = x.to(device), y.to(device)
#             logits, _ = model(x)
#             loss = criterion(logits, y)

#             batch_size = x.size(0)
#             total_loss += loss.item() * batch_size
#             total_samples += batch_size
#             # dice_fn = DiceLoss().to(device)
#         # iou = utils.miou_score(logits, target_train))
#             dice_score=utils.compute_modified_miou(logits, y)
#             total_iou += dice_score * batch_size


#             avg_loss = total_loss / total_samples
#             avg_iou = total_iou / total_samples
#             pbar.set_postfix(loss=avg_loss, miou=avg_iou)

#     avg_loss = total_loss / total_samples
#     avg_iou = total_iou / total_samples
#     return avg_loss, avg_iou
def train(loader, model, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    total_metrics = {'accuracy': 0, 'f1_macro': 0, 'precision': 0, 'recall': 0}
    total_samples = 0

    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"[Train Epoch {epoch}]")

    for num_batch, (x, y) in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        metrics = compute_metrics(logits, y)
        for key in total_metrics:
            total_metrics[key] += metrics[key] * batch_size

        avg_loss = total_loss / total_samples
        avg_f1 = total_metrics['f1_macro'] / total_samples
        pbar.set_postfix(loss=avg_loss, f1_macro=avg_f1)

        global_step = num_batch + epoch * len(loader)
        wandb.log({
            "train/avg_loss": loss.item(),
            "train/avg_f1_macro": metrics['f1_macro']
        }, step=global_step)

    avg_loss = total_loss / total_samples
    avg_metrics = {key: value / total_samples for key, value in total_metrics.items()}
    return avg_loss, avg_metrics

def evaluate(loader, model, criterion, device):
    model.eval()
    total_loss = 0
    total_metrics = {'accuracy': 0, 'f1_macro': 0, 'precision': 0, 'recall': 0}
    total_samples = 0

    pbar = tqdm(loader, desc="[Valid]")

    with torch.no_grad():
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = criterion(logits, y)

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            metrics = compute_metrics(logits, y)
            for key in total_metrics:
                total_metrics[key] += metrics[key] * batch_size

            avg_loss = total_loss / total_samples
            avg_f1 = total_metrics['f1_macro'] / total_samples
            pbar.set_postfix(loss=avg_loss, f1_macro=avg_f1)

    avg_loss = total_loss / total_samples
    avg_metrics = {key: value / total_samples for key, value in total_metrics.items()}
    return avg_loss, avg_metrics

if __name__ == '__main__':
    wandb.init(project="segmentation-darts", name=args.save)
    wandb.config.update(args)
    main()
