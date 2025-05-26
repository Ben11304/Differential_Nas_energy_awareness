import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from genotypes import PRIMITIVES
from genotypes import Genotype






class ClassificationHead(nn.Module):
    """
    Head cho bài toán phân loại, chuyển tensor [B, C_prev, H, W] thành [B, num_classes].
    Gồm: AdaptiveAvgPool2d -> Flatten -> Linear.
    """
    def __init__(self, C_prev, num_classes, dropout_prob=0.0):
        super(ClassificationHead, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)  # Giảm về [B, C_prev, 1, 1]
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else nn.Identity()
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, x):
        x = self.pool(x)  # [B, C_prev, 1, 1]
        x = x.view(x.size(0), -1)  # [B, C_prev]
        x = self.dropout(x)
        x = self.classifier(x)  # [B, num_classes]
        return x


class MixedOp(nn.Module):
    def __init__(self, C, stride, primitives):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self._energy_flags = []
        self._macs = []

        for primitive in primitives:  # Sử dụng primitives được truyền vào
            op = OPS[primitive](C, stride, False)
            self._ops.append(op)
            self._macs.append(torch.tensor(0.0, dtype=torch.float32))
            self._energy_flags.append(False)

    def forward(self, x, weights):
        total_macs = sum(w * op.energy for w, op in zip(weights, self._ops))
        result = sum(w * op(x) for w, op in zip(weights, self._ops))
        return result, total_macs

class Cell(nn.Module):
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, primitives):
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)

        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride, primitives)  # Truyền primitives riêng cho cell
                self._ops.append(op)

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        offset = 0
        total_cell_macs = 0

        for i in range(self._steps):
            new_state = 0
            for j, h in enumerate(states):
                op = self._ops[offset + j]
                out, macs = op(h, weights[offset + j])
                new_state += out
                total_cell_macs += macs
            offset += len(states)
            states.append(new_state)

        return torch.cat(states[-self._multiplier:], dim=1), total_cell_macs

class Network(nn.Module):
    def __init__(self, C, num_classes, primitive_sets, criterion,
                 steps=4, multiplier=4, stem_multiplier=3, device='cuda'):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self._device = device

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False

        # Khởi tạo cells với primitive_sets
        for i, primitives in enumerate(primitive_sets):
            if i in [len(primitive_sets) // 3, 2 * len(primitive_sets) // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(
                steps, multiplier,
                C_prev_prev, C_prev, C_curr,
                reduction, reduction_prev, primitives
            )
            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.head = ClassificationHead(C_prev=C_prev, num_classes=num_classes).to(self._device)
        self._initialize_alphas(primitive_sets)
        self.to(self._device)

    def new(self):
        model_new = Network(
            self._C, self._num_classes, self._primitive_sets, self._criterion,
            self._steps, self._multiplier, device=self._device
        )
        model_new.to(self._device)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        for cell_new in model_new.cells:
            for mixed_op_new in cell_new._ops:
                for op_new in mixed_op_new._ops:
                    op_new.energy_flag = 1
        return model_new

    def forward(self, input):
        input = input.to(self._device)
        s1 = self.stem(input)
        total_macs = 0
        s0 = s1
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = F.softmax(self.alphas_reduce[i], dim=-1)
            else:
                weights = F.softmax(self.alphas_normal[i], dim=-1)
            s0, s1_mixed = s1, cell(s0, s1, weights)
            s1 = s1_mixed[0]
            total_macs += s1_mixed[1]
        logits = self.head(s1)
        return logits, total_macs

    def _loss(self, input, target):
        input = input.to(self._device)
        target = target.to(self._device)
        logits, _ = self(input)
        return self._criterion(logits, target)

    def _initialize_alphas(self, primitive_sets):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        self.alphas_normal = []
        self.alphas_reduce = []
        
        # Khởi tạo alpha riêng cho mỗi cell dựa trên primitive_sets
        for primitives in primitive_sets:
            num_ops = len(primitives)
            alpha_normal = nn.Parameter(1e-3 * torch.randn(k, num_ops, device=self._device))
            alpha_reduce = nn.Parameter(1e-3 * torch.randn(k, num_ops, device=self._device))
            self.alphas_normal.append(alpha_normal)
            self.alphas_reduce.append(alpha_reduce)

        self._arch_parameters = []
        for i in range(len(primitive_sets)):
            self._arch_parameters.append(self.alphas_normal[i])
            self._arch_parameters.append(self.alphas_reduce[i])
        self._primitive_sets = primitive_sets  # Lưu lại để dùng trong genotype

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):
        def _parse(weights, primitives):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(
                    range(i + 2),
                    key=lambda x: -max(
                        W[x][k] for k in range(len(W[x])) if 'none' not in primitives or k != primitives.index('none')
                    )
                )[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if 'none' in primitives and k != primitives.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                        elif 'none' not in primitives:
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((primitives[k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = []
        gene_reduce = []
        for i, primitives in enumerate(self._primitive_sets):
            gene_normal += _parse(F.softmax(self.alphas_normal[i], dim=-1).detach().cpu().numpy(), primitives)
            gene_reduce += _parse(F.softmax(self.alphas_reduce[i], dim=-1).detach().cpu().numpy(), primitives)

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype