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
import copy
from tqdm import tqdm  # Thêm vào đầu file nếu chưa có
import wandb
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")



# from torch.autograd import Variable  # Đã không còn cần thiết
from model_search import Network
from architect import Architect


parser = argparse.ArgumentParser("cifar")
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
parser.add_argument('--layers', type=int, default=3, help='total number of layers')
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


args.save = './log/'  
os.makedirs(args.save, exist_ok=True) 

# Tạo tên file log với timestamp
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


Agricultural_classes=8

import torch
import os
from torch.utils.data import Dataset
import glob

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


def main():

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device=args.device
    logging.info("args = %s", args)
    criterion= nn.BCEWithLogitsLoss().to(device)
    
    model = Network(args.init_channels, Agricultural_classes , args.layers, criterion, device=args.device, snn_step=args.snn_step)
    model.to(device)

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    full_dataset = CustomPTSegmentationDataset(root_dir="../../../Agriculture-Vision-2021_processed_zip/trainrandcrop256/")
    indices = list(range(5000))
    np.random.shuffle(indices)
    train_indices = indices[:2500]
    val_indices = indices[4500:5000]

    train_queue = torch.utils.data.DataLoader(full_dataset, batch_size=args.batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(train_indices), num_workers=2)
    valid_queue = torch.utils.data.DataLoader(full_dataset, batch_size=args.batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(val_indices), num_workers=2)

    # train_queue = torch.utils.data.DataLoader(
    #     train_data, batch_size=args.batch_size,
    #     sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
    #     pin_memory=True, num_workers=2
    # )

    # valid_queue = torch.utils.data.DataLoader(
    #     train_data, batch_size=args.batch_size,
    #     sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
    #     pin_memory=True, num_workers=2
    # )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min
    )
    initial_weights = copy.deepcopy(model.state_dict())
    architect = Architect(model, args, device=args.device)
    wandb.init(project="nas-search", name=f"search-{time.strftime('%Y%m%d-%H%M%S')}")
    wandb.config.update(args)

    for epoch in range(args.epochs):
        # Bắt đầu mỗi epoch, cập nhật scheduler
        scheduler.step()
        # Với PyTorch mới, nên dùng get_last_lr()
        lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else scheduler.get_lr()[0]

        logging.info('epoch %d lr %e', epoch, lr)

        genotype = model.genotype()
        logging.info('genotype = %s', genotype)

        # In ma trận alpha
        logging.info(F.softmax(model.alphas_normal, dim=-1))
        logging.info(F.softmax(model.alphas_reduce, dim=-1))

        # training
        train_acc,train_F1, train_obj , macs = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, device,args)
        macs=float(macs)
        logging.info("train mIoU: %.4f | train F1: %.4f  | Loss: %.4f | Avg MACs: %s", train_acc,train_F1, train_obj , macs)

        # validation
        valid_acc,valid_F1, valid_obj, valid_macs = infer(valid_queue, model, criterion, device)
        # valid_macs=format_macs(valid_macs)
        logging.info("Validation mIoU: %.4f | Val F1: %.4f  | Loss: %.4f | Avg MACs: %s", valid_acc,valid_F1,  valid_obj, valid_macs)
        wandb.log({
            "epoch/train_mIoU": train_acc,
            "epoch/train_loss": train_obj,
            "epoch/train_F1":train_F1,
            "epoch/val_mIoU": valid_acc,
            "epoch/val_loss": valid_obj,
            "epoch/valid_F1":valid_F1,
            "epoch/train_energy": macs,
            "epoch": epoch
        })

        utils.save(model, os.path.join(args.save, 'weights.pt'))
        # utils.reset_weights_only(model, initial_weights)
        
def compute_metrics(preds, labels, threshold=0.5):
    preds = torch.sigmoid(preds) > threshold
    labels = labels > 0.2
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

def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, device, args):
    objs = utils.AvgrageMeter()
    metrics_avg = {'accuracy': utils.AvgrageMeter(), 'f1_macro': utils.AvgrageMeter()}

    model.train()
    valid_iter = iter(valid_queue)
    pbar = tqdm(train_queue, desc="[Train]", leave=False) # trọng số cho phần mac penalty

    for step, (input_train, target_train) in enumerate(pbar):
        n = input_train.size(0)

        input_train = input_train.to(device)
        target_train = target_train.to(device)

        try:
            input_search, target_search = next(valid_iter)
        except StopIteration:
            valid_iter = iter(valid_queue)
            input_search, target_search = next(valid_iter)

        input_search = input_search.to(device)
        target_search = target_search.to(device)

        # Architect step
        architect.step(input_train, target_train, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

        optimizer.zero_grad()
        logits, energy = model(input_train)


        loss_task = criterion(logits, target_train)

    
        loss_total=loss_task
        loss_total.backward()

        optimizer.step()
        metrics = compute_metrics(logits, target_train)
        objs.update(loss_total.item(), n)
        metrics_avg['accuracy'].update(metrics['accuracy'], n)
        metrics_avg['f1_macro'].update(metrics['f1_macro'], n)

        wandb.log({
            "train/step_loss": loss_total.item(),
            "train/step_accuracy": metrics['accuracy'],
            "train/step_f1_macro": metrics['f1_macro'],
            "train/step_energy": energy,
            "train/lr": lr
        })

        pbar.set_postfix({
            "loss_task": objs.avg,
            "accuracy": metrics_avg['accuracy'].avg,
            "f1_macro": metrics_avg['f1_macro'].avg,
            "energy": f"{energy}"
        })

    return metrics_avg['accuracy'].avg, metrics_avg['f1_macro'].avg, objs.avg, energy





def infer(valid_queue, model, criterion, device):
    objs = utils.AvgrageMeter()
    metrics_avg = {'accuracy': utils.AvgrageMeter(), 'f1_macro': utils.AvgrageMeter()}

    model.eval()

    pbar = tqdm(valid_queue, desc="[Valid]", leave=False)

    with torch.no_grad():
        for step, (input_valid, target_valid) in enumerate(pbar):
            input_valid = input_valid.to(device)
            target_valid = target_valid.to(device)

            logits,energy = model(input_valid)
            loss = criterion(logits, target_valid)

            metrics = compute_metrics(logits, target_valid)
            n = input_valid.size(0)
            objs.update(loss.item(), n)
            metrics_avg['accuracy'].update(metrics['accuracy'], n)
            metrics_avg['f1_macro'].update(metrics['f1_macro'], n)

            pbar.set_postfix({
                "val_loss": objs.avg,
                "val_accuracy": metrics_avg['accuracy'].avg,
                "val_f1_macro": metrics_avg['f1_macro'].avg,
                "total_mac": energy
            })

    return metrics_avg['accuracy'].avg, metrics_avg['f1_macro'].avg, objs.avg, energy

def format_macs(value):
    if value >= 1e9:
        return f"{value / 1e9:.2f} G"
    elif value >= 1e6:
        return f"{value / 1e6:.2f} M"
    elif value >= 1e3:
        return f"{value / 1e3:.2f} K"
    elif value >= 1:
        return f"{value:.2f}"
    elif value >= 1e-3:
        return f"{value * 1e3:.2f} e-3"
    elif value >= 1e-6:
        return f"{value * 1e6:.2f} e-6"
    else:
        return f"{value:.2e}"




if __name__ == '__main__':
    main()
