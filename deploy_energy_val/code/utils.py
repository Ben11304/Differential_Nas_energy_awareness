import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.functional as F
import math

from ptflops import get_model_complexity_info

def compute_bce_loss(predict, target):
    """
    predict: Tensor [B, C, H, W] - logits (chưa sigmoid)
    target: Tensor [B, C, H, W] - binary masks (0 hoặc 1)
    """
    # Step 1: Sigmoid để chuyển logits -> xác suất
    probs = torch.sigmoid(predict)

    # Step 2: Tính Binary Cross Entropy
    bce_loss = F.binary_cross_entropy(probs, target.float(), reduction='mean')

    return bce_loss


def reset_weights_only(model, initial_weights):
    with torch.no_grad():
        current_state = model.state_dict()
        for k, v in initial_weights.items():
            # Bỏ qua alpha (tham số kiến trúc) để giữ nguyên
            if "alphas" not in k:
                current_state[k].copy_(v)


class BCE_Dice_LovaszLoss(nn.Module):
    def __init__(self, weight_bce=1.0, weight_dice=1.0, weight_lovasz=1.0):
        super(BCE_Dice_LovaszLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.weight_lovasz = weight_lovasz

    def forward(self, logits, targets):
        """
        logits: (B, C, H, W)
        targets: (B, C, H, W)
        """
        bce_loss = self.bce(logits, targets)

        dice_loss = self.dice_loss(logits, targets)

        lovasz_loss = self.lovasz_hinge_loss(logits, targets)

        total_loss = (self.weight_bce * bce_loss +
                      self.weight_dice * dice_loss +
                      self.weight_lovasz * lovasz_loss)

        return total_loss

    def dice_loss(self, logits, targets, eps=1e-6):
        probs = torch.sigmoid(logits)
        targets = targets.float()
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2 * intersection + eps) / (union + eps)
        dice_loss = 1 - dice.mean()
        return dice_loss

    def lovasz_hinge_loss(self, logits, labels):
        # Flatten predictions and labels
        logits_flat = logits.view(-1)
        labels_flat = labels.view(-1)
        labels_flat = labels_flat.float()

        signs = 2.0 * labels_flat - 1.0
        errors = (1.0 - logits_flat * signs)

        errors_sorted, perm = torch.sort(errors, descending=True)
        perm = perm.data
        gt_sorted = labels_flat[perm]

        grad = self.lovasz_grad(gt_sorted)
        loss = torch.dot(F.relu(errors_sorted), grad)

        return loss

    def lovasz_grad(self, gt_sorted):
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1.0 - intersection / union
        if p > 1:
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard




def combined_bce_dice_loss(preds, targets, alpha=0.5, smooth=1e-6):
    """
    Compute Combined BCE + Dice Loss for multi-label semantic segmentation.
    
    Args:
        preds (torch.Tensor): Predicted logits [B, C, H, W] or [C, H, W].
        targets (torch.Tensor): Ground truth binary labels [B, C, H, W] or [C, H, W].
        alpha (float): Weight for BCE loss (1-alpha for Dice). Default: 0.5.
        smooth (float): Smoothing factor to avoid division by zero in Dice. Default: 1e-6.
    
    Returns:
        torch.Tensor: Combined loss value.
    """
    # Ensure inputs are float and have consistent shapes
    preds = preds.float()
    targets = targets.float()
    
    # BCE Loss
    bce_loss = nn.BCEWithLogitsLoss()(preds, targets)
    
    # Dice Loss
    preds_sig = torch.sigmoid(preds)  # Convert logits to probabilities
    preds_flat = preds_sig.flatten()  # Flatten for Dice computation
    targets_flat = targets.flatten()
    
    intersection = (preds_flat * targets_flat).sum()
    dice_loss = 1 - (2. * intersection + smooth) / (preds_flat.sum() + targets_flat.sum() + smooth)
    
    # Combined Loss
    return alpha * bce_loss + (1 - alpha) * dice_loss

class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt



# def compute_modified_miou(pred_tensor, gt_tensor, num_classes=9):
#     '''Compute mean Intersection over Union (mIoU) for multi-class segmentation.
#     Args: 
#         pred_tensor (torch.Tensor): Predicted segmentation map of shape (N, C, H, W) or (C, H, W).
#         gt_tensor (torch.Tensor): Ground truth segmentation map of shape (N, C, H, W) or (C, H, W).
#         num_classes (int): Number of classes including background.
#     Returns:
#         float: Mean Intersection over Union (mIoU) score.'''
#     if pred_tensor.shape[0] != gt_tensor.shape[0]:
#         raise ValueError(f"Batch size mismatch: pred {pred_tensor.shape[0]} vs gt {gt_tensor.shape[0]}")
#     if gt_tensor.shape[1] != num_classes:
#         raise ValueError(f"Expected {num_classes - 1} channels in gt_tensor, got {gt_tensor.shape[1]}")
#     N, _, H, W = gt_tensor.shape
#     device = gt_tensor.device
#     total_pixels = N * H * W

#     gt_flat_with_bg=gt_tensor.view(N, C, -1).permute(0, 2, 1).reshape(-1, C)
#     pred_flat_with_bg=pred_tensor.view(N, C, -1).permute(0, 2, 1).reshape(-1, C)
#   # Shape: [N*H*W, num_classes-1]
#     pred_softmax = torch.softmax(pred_tensor, dim=1).view(total_pixels, num_classes)  # Shape: [N*H*W, num_classes-1]
#     pred_flat_with_bg = torch.argmax(pred_softmax, dim=1, keepdim=True).float() # Apply threshold
#     # background_gt = (gt_flat.sum(dim=1, keepdim=True) == 0).long()  # 1 if no class is present
#     # gt_flat_with_bg = torch.cat([gt_flat, background_gt], dim=1)
#     print(gt_tensor.size())
#     print(gt_flat_with_bg.size())
#     for pixel in gt_flat_with_bg:
#         print( pixel.size())
#     if torch.any(gt_flat_with_bg.sum(dim=1) < 1e-6):
#         raise ValueError("Some pixels in gt_flat_with_bg have no labels, even after adding background")


#     # background_pred = (pred_binary.sum(dim=1, keepdim=True) == 0).float()  # 1 if no class > 0.5
#     # pred_flat_with_bg = torch.cat([pred_softmax, background_pred], dim=1)  # Shape: [N*H*W, num_classes]
#     if torch.any(pred_flat_with_bg.sum(dim=1) < 1e-6):
#         raise ValueError("Some pixels in pred_flat_with_bg have no labels, even after adding background")

#     # Convert predictions to labels
#     pred_labels = torch.argmax(pred_flat_with_bg, dim=1)  # Shape: [N*H*W]
#     valid_pred = (pred_labels >= 0) & (pred_labels < num_classes)

#     # Confusion matrix
#     confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
#     for i in range(total_pixels):
#         if not valid_pred[i]:
#             print(f"pixel {i} is invalid")
#             continue
#         y_indices = torch.nonzero(gt_flat_with_bg[i], as_tuple=False).squeeze()
#         if y_indices.numel() == 0:
#             print(f"pixel {i} gt_flat_with_bg: {gt_flat_with_bg[i]}, sum: {gt_flat_with_bg[i].sum().item()}")
#             continue
#         x = pred_labels[i].item()
#         if y_indices.dim() == 0:
#             y = y_indices.item()
#             if x == y:
#                 confusion_matrix[y, y] += 1
#             else:
#                 confusion_matrix[x, y] += 1
#         else:
#             for y in y_indices:
#                 y = y.item()
#                 if x == y:
#                     confusion_matrix[y, y] += 1
#                 else:
#                     confusion_matrix[x, y] += 1

#     # Compute mIoU
#     ious = []
#     for c in range(num_classes):
#         tp = confusion_matrix[c, c]
#         pred_sum = confusion_matrix[:, c].sum()
#         gt_sum = confusion_matrix[c, :].sum()
#         union = pred_sum + gt_sum - tp
#         ious.append((tp / union).item() if union > 0 else 0.0)
#     print(f"total iou {ious}")
    
#     return np.mean(ious)



import torch
import numpy as np

def compute_modified_miou(pred_tensor, gt_tensor, num_classes=9):
    '''Compute mean Intersection over Union (mIoU) for multi-class segmentation with multi-label ground truth.
    Args: 
        pred_tensor (torch.Tensor): Predicted segmentation map of shape (N, C, H, W) or (C, H, W).
        gt_tensor (torch.Tensor): Ground truth segmentation map of shape (N, C, H, W) or (C, H, W).
        num_classes (int): Total number of classes including background.
    Returns:
        float: Mean Intersection over Union (mIoU) score.
    '''
    # Handle single sample case
    if pred_tensor.dim() == 3:
        pred_tensor = pred_tensor.unsqueeze(0)
        gt_tensor = gt_tensor.unsqueeze(0)

    # Validate inputs
    if pred_tensor.shape[0] != gt_tensor.shape[0]:
        raise ValueError(f"Batch size mismatch: pred {pred_tensor.shape[0]} vs gt {gt_tensor.shape[0]}")
    if gt_tensor.shape[1] != num_classes:
        raise ValueError(f"Expected {num_classes} channels in gt_tensor, got {gt_tensor.shape[1]}")

    N, C, H, W = gt_tensor.shape
    device = gt_tensor.device
    total_pixels = N * H * W

    # Flatten tensors
    gt_flat = gt_tensor.view(N, C, -1).permute(0, 2, 1).reshape(total_pixels, num_classes)  # Shape: [N*H*W, num_classes]
    
    # Apply weights before softmax
    weights = torch.ones(C, device=device)
    weights[:num_classes-1] = 3.0  # Multiply non-background classes (0 to 8) by 3
    weights[num_classes-1] = 0.05  # Multiply background (class 9) by 0.95
    weighted_pred = pred_tensor * weights.view(1, -1, 1, 1)  # Apply weights to pred_tensor
    print(f"Logits trung bình mỗi lớp: {pred_tensor.mean(dim=[0,2,3])}")
    pred_softmax = torch.softmax(weighted_pred, dim=1)
    pred_flat = pred_softmax.view(N, C, -1).permute(0, 2, 1).reshape(total_pixels, num_classes)  # Shape: [N*H*W, num_classes]
    pred_labels = torch.argmax(pred_flat, dim=1)  # Shape: [N*H*W], predicted class indices
    print(f"Phân bố lớp dự đoán: {torch.bincount(pred_labels)}")
    # Check for valid ground truth
    if torch.any(gt_flat.sum(dim=1) < 1e-6):
        raise ValueError("Some pixels in gt_flat have no labels")

    # Initialize confusion matrix
    confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)

    # Case 1: Prediction matches a ground truth label
    gt_nonzero = gt_flat > 0  # Shape: [N*H*W, num_classes], binary mask of ground truth labels
    pred_matches = gt_nonzero[torch.arange(total_pixels), pred_labels]  # Shape: [N*H*W], True if pred_labels[i] matches one of the gt labels
    for c in range(num_classes):
        matched_and_gt_c = pred_matches & gt_nonzero[:, c]  # Shape: [N*H*W]
        confusion_matrix[c, c] += matched_and_gt_c.sum().item()

    # Case 2: Prediction does not match any ground truth label
    unmatched_pixels = ~pred_matches  # Shape: [N*H*W], True if pred_labels[i] does not match gt
    for x in range(num_classes):
        pred_is_x = (pred_labels == x) & unmatched_pixels  # Shape: [N*H*W]
        for y in range(num_classes):
            increment = (pred_is_x & gt_nonzero[:, y]).sum().item()
            confusion_matrix[x, y] += increment

    # Compute IoU for each class
    ious = []
    gt_sums = []
    for c in range(num_classes):
        tp = confusion_matrix[c, c]
        pred_sum = confusion_matrix[c, :].sum()
        gt_sum = confusion_matrix[:, c].sum()
        union = pred_sum + gt_sum - tp
        iou = (tp / union).item() if union > 0 else 0.0
        ious.append(iou)
        gt_sums.append(gt_sum.item())

    print(f"Per-class IoU: {ious}")
    print(f"Ground truth pixel counts per class: {gt_sums}")
    mIoU = np.mean(ious)
    # print(f"Computed mIoU (simple mean): {mIoU}")
    return mIoU


def count_module_macs(module, data_dims) -> int:
    """
    Computes the MACs for a given PyTorch module.

    Args:
        module (torch.nn.Module): the PyTorch module for which the MACs should be counted.
        data_dims (tuple): the dimensions of the input data (batch_size, channels, height, width).

    Returns:
        int: number of MACs for the given module.
    """
    
    # BatchNorm2d doesn't contribute to MACs
    if isinstance(module, torch.nn.BatchNorm2d):
        print("BatchNorm2d detected")
        return 0

    # Manually calculate for MaxPool2d
    if isinstance(module, torch.nn.MaxPool2d):
        s = module.stride
        k = module.kernel_size
        p = module.padding

        # Ensure kernel size and stride are integers
        k = k if isinstance(k, int) else k[0]
        s = s if isinstance(s, int) else s[0]
        p = p if isinstance(p, int) else p[0]

        out_h = math.floor(((data_dims[2] - k + (2 * p)) / s) + 1)
        out_w = math.floor(((data_dims[3] - k + (2 * p)) / s) + 1)
        flops = k * k * out_h * out_w * data_dims[1] * data_dims[0]
        macs = flops / 2  # MaxPool uses FLOPs not MACs directly
        return int(macs)

    # General case: use ptflops
    try:
        macs, params = get_model_complexity_info(
            module, tuple(data_dims[1:]), as_strings=False, print_per_layer_stat=False
        )
        macs = 0 if macs is None else macs
    except Exception as e:
        print(f"Error getting MACs for module {module.__class__.__name__}: {e}")
        macs = 0
    return int(macs * data_dims[0])



def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # cách 1: dùng contiguous() trước view
        # correct_k = correct[:k].contiguous().view(-1).float().sum(0)

        # hoặc cách 2: dùng reshape(-1) thay thế
        correct_k = correct[:k].reshape(-1).float().sum(0)

        res.append(correct_k.mul_(100.0 / batch_size))
    return res



class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))


# def drop_path(x, drop_prob):
#   if drop_prob > 0.:
#     keep_prob = 1.-drop_prob
#     mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
#     x.div_(keep_prob)
#     x.mul_(mask)
#   return x
def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1. - drop_prob
    # Tạo mask trên cùng device và dtype với x
    mask = torch.empty(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device).bernoulli_(keep_prob)
    mask = Variable(mask)
    x.div_(keep_prob)
    x.mul_(mask)
  return x

def iou_score(outputs, masks, threshold=0.5):
    outputs = torch.sigmoid(outputs) > threshold
    intersection = (outputs & masks.bool()).float().sum((2, 3))
    union = (outputs | masks.bool()).float().sum((2, 3))
    iou = (intersection / (union + 1e-6)).mean()
    return iou.item()


def miou_score(outputs, targets, threshold=0.5, eps=1e-6):
    """
    outputs: raw model logits (before sigmoid), shape [B, C, H, W]
    targets: binary ground truth masks, shape [B, C, H, W]
    """
    # Apply sigmoid and binarize
    outputs = (torch.sigmoid(outputs) > threshold).float()
    targets = (targets > 0.5).float()  # Ensure binary
    # print("Output stats:", outputs.min().item(),outputs.max().item(), outputs.mean().item())
    # print("targets stats:", targets.min().item(),targets.max().item(), targets.mean().item())

    # Intersection and union per sample, per class
    intersection = (outputs * targets).sum(dim=(2, 3))  # [B, C]
    union = ((outputs + targets) > 0).float().sum(dim=(2, 3))  # [B, C]

    # IoU per class, per sample
    iou = (intersection + eps) / (union + eps)  # [B, C]

    # mIoU = mean over all classes and all samples
    return iou.mean().item()




def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

