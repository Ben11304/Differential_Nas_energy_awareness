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
import copy
from tqdm import tqdm 
import wandb
import warnings
import torch
import os
from torch.utils.data import Dataset
import glob

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
parser.add_argument('--rate', type=tuple, default=[1,0], help='weight decay for arch encoding')
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

CIFAR_CLASSES = 10


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
        label = label[:8]
        if self.transform:
            image, label = self.transform(image, label)
        return image, label


def main():

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device=args.device
    logging.info("args = %s", args)
    criterion = nn.CrossEntropyLoss().to(device)


    #chỉnh sửa primitive huấn luyện tại đây
    primitive_sets = [
    ['conv_3x3', 'sep_conv_3x3','dil_conv_5x5','max_pool_3x3'],  # Cell 1
    ['max_pool_3x3', 'skip_connect'],  # Cell 2
    ]



    model = Network(args.init_channels, CIFAR_CLASSES , primitive_sets, criterion, device=args.device)
    model.to(device)

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

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
        lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else scheduler.get_lr()[0]

        logging.info('epoch %d lr %e', epoch, lr)

        genotype = model.genotype()
        logging.info('genotype = %s', genotype)

        # In ma trận alpha
        # Log softmax of each alpha tensor in the lists
        logging.info("Softmax of alphas_normal:")
        for i, alpha in enumerate(model.alphas_normal):
            logging.info(f"Cell {i}: {F.softmax(alpha, dim=-1)}")

        logging.info("Softmax of alphas_reduce:")
        for i, alpha in enumerate(model.alphas_reduce):
            logging.info(f"Cell {i}: {F.softmax(alpha, dim=-1)}")

        train_top1, train_top5, train_objs, energy =train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, device,args)
        logging.info("train Top 1: %.4f | train Top 5: %.4f  | Loss: %.4f | Total energy: %s", train_top1, train_top5, train_objs, energy)

        valid_top1, valid_top5, valid_objs, =infer(valid_queue, model, criterion, device)
        
        logging.info("Validation Top 1: %.4f | Val Top 5: %.4f  | Loss: %.4f",valid_top1,  valid_top5, valid_objs)
        wandb.log({
            "epoch/train_Top1": train_top1,
            "epoch/train_loss": train_objs,
            "epoch/train_Top5":train_top5,
            "epoch/val_Top1": valid_top1,
            "epoch/val_loss": valid_objs,
            "epoch/valid_top5":valid_top5,
            "epoch/train_energy": energy,
            "epoch": epoch
        })

        utils.save(model, os.path.join(args.save, 'weights.pt'))
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
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    model.train()
    valid_iter = iter(valid_queue)
    pbar = tqdm(train_queue, desc="[Train]", leave=False) 

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
        architect.step(input_train, target_train, input_search, target_search, lr, optimizer, args.unrolled)

        
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
            "train/lr": lr
        })
        pbar.set_postfix({
            "loss_task": objs.avg,
            "accuracy": top5.avg,
            "energy": f"{energy}"
        })
    return top1.avg, top5.avg, objs.avg, energy





def infer(valid_queue, model, criterion, device):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    pbar = tqdm(valid_queue, desc="[Valid]", leave=False)

    with torch.no_grad():
        for step, (input_valid, target_valid) in enumerate(pbar):
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
    main()
