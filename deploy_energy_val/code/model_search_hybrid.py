"""
Hybrid NAS model = pretrained ResNet-18 backbone (frozen) + DART cells on top.
Energy is computed only for the NAS cells (the backbone is fixed so its cost is
constant across the sweep and does not influence alpha gradients).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

from model_search import Cell, ClassificationHead


class ResNetStem(nn.Module):
    """ResNet-18 truncated to produce a feature map with ~128 channels."""

    def __init__(self, out_channels, freeze=True, cut='layer3'):
        super().__init__()
        net = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        modules = [net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2]
        backbone_channels = 128
        if cut == 'layer3':
            modules.append(net.layer3)
            backbone_channels = 256
        self.backbone = nn.Sequential(*modules)
        self.project = nn.Sequential(
            nn.Conv2d(backbone_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        # Keep backbone in eval mode so BN statistics don't drift when frozen
        self.backbone.eval()
        return self

    def forward(self, x):
        with torch.no_grad():
            feat = self.backbone(x)
        return self.project(feat)


class HybridNetwork(nn.Module):
    """ResNet backbone + DART mixed-op cells + classification head."""

    def __init__(self, C, num_classes, primitive_sets, criterion,
                 steps=4, multiplier=4, backbone_cut='layer3',
                 freeze_backbone=True, device='cuda'):
        super().__init__()
        self._C = C
        self._num_classes = num_classes
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self._device = device

        self.stem = ResNetStem(C, freeze=freeze_backbone, cut=backbone_cut)

        C_prev_prev, C_prev, C_curr = C, C, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i, primitives in enumerate(primitive_sets):
            reduction = False  # see fix in model_search.py
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr,
                        reduction, reduction_prev, primitives)
            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.head = ClassificationHead(C_prev=C_prev, num_classes=num_classes).to(device)
        self._initialize_alphas(primitive_sets)
        self.to(device)

    def forward(self, x):
        x = x.to(self._device)
        s1 = self.stem(x)
        s0 = s1
        # Paper B FLOPs-swap: `e` from MixedOp is now exact per-sample MACs
        # (was sklearn-estimated energy). Name kept for API stability;
        # architect.py's max_seen normalization applies unchanged.
        total_energy = 0
        for i, cell in enumerate(self.cells):
            weights = F.softmax(self.alphas_normal[i], dim=-1)
            out, e = cell(s0, s1, weights)
            s0, s1 = s1, out
            total_energy = total_energy + e
        return self.head(s1), total_energy

    def _loss(self, input, target):
        logits, _ = self(input)
        return self._criterion(logits, target.to(self._device))

    def _initialize_alphas(self, primitive_sets):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        self.alphas_normal = []
        self.alphas_reduce = []
        for primitives in primitive_sets:
            num_ops = len(primitives)
            a = nn.Parameter(1e-3 * torch.randn(k, num_ops, device=self._device))
            b = nn.Parameter(1e-3 * torch.randn(k, num_ops, device=self._device))
            self.alphas_normal.append(a)
            self.alphas_reduce.append(b)
        self._arch_parameters = []
        for i in range(len(primitive_sets)):
            self._arch_parameters.append(self.alphas_normal[i])
            self._arch_parameters.append(self.alphas_reduce[i])
        self._primitive_sets = primitive_sets

    def arch_parameters(self):
        return self._arch_parameters

    def new(self):
        """Required by Architect._construct_model_from_theta (unrolled case)."""
        model_new = HybridNetwork(
            self._C, self._num_classes, self._primitive_sets, self._criterion,
            self._steps, self._multiplier, device=self._device,
        )
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def genotype(self):
        # Reuse the same parser as Network
        from model_search import Network
        return Network.genotype(self)
