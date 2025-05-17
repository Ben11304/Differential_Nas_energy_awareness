import torch
import torch.nn as nn
from operations import *
# from torch.autograd import Variable  # Không còn cần thiết
from utils import drop_path

class SNNHead(nn.Module):
    """
    SNN head sử dụng multi-step spiking, gồm Conv2d (1x1) và Upsampling.
    Đầu ra trung bình spike sau num_steps để ổn định kết quả.
    """
    def __init__(self, C_prev, num_classes, num_steps=1, beta=0.9, scale_factor=4):
        super(SNNHead, self).__init__()
        self.num_steps = num_steps
        self.scale_factor = scale_factor
        
        # Convolution 1x1 chuyển channels về num_classes
        self.conv = nn.Conv2d(C_prev, num_classes, kernel_size=1, bias=False)
        
        # Neuron Leaky Integrate-and-Fire
        self.lif = snn.Leaky(beta=beta, learn_beta=False)
        
        # Upsample để khôi phục kích thước ban đầu
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)

    def forward(self, x):
        mem = self.lif.init_leaky()
        spk_acc = 0
        
        for _ in range(self.num_steps):
            cur = self.conv(x)  # hoặc cur = torch.tanh(cur)
            spk, mem = self.lif(cur, mem)
            mem = torch.clamp(mem, -5.0, 5.0)
            spk= torch.clamp(spk, -1.0, 1.0)
            spk_acc += spk
            mem.detach()
        
        # Trung bình spike qua các bước thời gian
        out = spk_acc / self.num_steps
        
        # Upsample
        out = self.upsample(out)
        return out


class ConvHead(nn.Module):
    """
    Head dùng Conv2D bình thường để thay thế SNNHead.
    Gồm Conv 1x1 + Upsampling.
    """
    def __init__(self, C_prev, num_classes, scale_factor=4):
        super(ConvHead, self).__init__()
        self.conv = nn.Conv2d(C_prev, num_classes, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(num_classes)
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.upsample(x)
        return x


class Cell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        print(C_prev_prev, C_prev, C)

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops.append(op)
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.:
                # drop_path chỉ áp dụng nếu op không phải Identity
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states.append(s)
        return torch.cat([states[i] for i in self._concat], dim=1)


class AuxiliaryHeadCIFAR(nn.Module):
    """
    Giả định input size 8x8
    """
    def __init__(self, C, num_classes):
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),  # image size = 2 x 2
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class AuxiliaryHeadImageNet(nn.Module):
    """
    Giả định input size 14x14
    """
    def __init__(self, C, num_classes):
        super(AuxiliaryHeadImageNet, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            # Theo code gốc, batchnorm cho 768 được comment vì lý do "paper consistency"
            # nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


# class NetworkCIFAR(nn.Module):
#     def __init__(self, C, num_classes, layers, auxiliary, genotype):
#         super(NetworkCIFAR, self).__init__()
#         self._layers = layers
#         self._auxiliary = auxiliary

#         stem_multiplier = 3
#         C_curr = stem_multiplier * C
#         self.stem = nn.Sequential(
#             nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
#             nn.BatchNorm2d(C_curr)
#         )

#         C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
#         self.cells = nn.ModuleList()
#         reduction_prev = False

#         for i in range(layers):
#             if i in [layers // 3, 2 * layers // 3]:
#                 C_curr *= 2
#                 reduction = True
#             else:
#                 reduction = False
#             cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
#             reduction_prev = reduction
#             self.cells.append(cell)
#             C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr

#             # Xác định kênh cho tầng auxiliary
#             if i == 2 * layers // 3:
#                 C_to_auxiliary = C_prev

#         if auxiliary:
#             self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)

#         self.global_pooling = nn.AdaptiveAvgPool2d(1)
#         self.classifier = nn.Linear(C_prev, num_classes)

#     def forward(self, x):
#         logits_aux = None
#         s0 = s1 = self.stem(x)
#         for i, cell in enumerate(self.cells):
#             s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
#             if i == 2 * self._layers // 3:
#                 if self._auxiliary and self.training:
#                     logits_aux = self.auxiliary_head(s1)

#         out = self.global_pooling(s1)
#         logits = self.classifier(out.view(out.size(0), -1))
#         return logits, logits_aux

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


class SimpleCNN(nn.Module):
    def __init__(self, C_prev, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(C_prev, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # Giả sử kích thước ảnh đầu vào là 256x256
        self.fc1 = nn.Linear(32 * 64 * 64, 128)  # Sau 2 lần pooling: 256 -> 128 -> 64
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 64 * 64)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # Logits đầu ra
        return x

class NetworkCIFAR(nn.Module):
    def __init__(self, C, num_classes, layers, auxiliary, genotype, input_channels=4, _snn_step=1,device="cpu"):
        super(NetworkCIFAR, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary
        self.drop_path_prob = 0.0
        self._snn_step=_snn_step
        self._device=device
        stem_multiplier = 3
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, C_curr, 3, padding=1),
            nn.BatchNorm2d(C_curr)
        )


        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False

        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr

            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
        # self.head = SNNHead(
        #         C_prev=C_prev, 
        #         num_classes=num_classes, 
        #         num_steps=self._snn_step,   # Có thể tùy chỉnh số bước thời gian
        #         beta=0.95,      # Có thể tùy chỉnh hệ số beta của neuron LIF
        #         scale_factor=4  # Tương tự như cũ
        #         ).to(self._device)


        # self.head = ConvHead(C_prev=C_prev, num_classes=num_classes, scale_factor=4).to(self._device)
        self.head= ClassificationHead(C_prev=C_prev, num_classes=num_classes).to(self._device)
    
        # self.global_pooling = nn.AdaptiveAvgPool2d(1)
        # self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, x):
        logits_aux = None
        s0 = s1 = self.stem(x)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)

        # out = self.global_pooling(s1)
        # logits = self.classifier(out.view(out.size(0), -1))
        # return logits, logits_aux
        logits=self.head(s1)
        return logits,logits_aux



class NetworkImageNet(nn.Module):
    def __init__(self, C, num_classes, layers, auxiliary, genotype):
        super(NetworkImageNet, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary

        self.stem0 = nn.Sequential(
            nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        self.stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )
        self.head = SNNHead(
            C_prev=C_prev, 
            num_classes=num_classes, 
            num_steps=self._snn_step,   # Có thể tùy chỉnh số bước thời gian
            beta=0.95,      # Có thể tùy chỉnh hệ số beta của neuron LIF
            scale_factor=4  # Tương tự như cũ
            ).to(self._device)

        C_prev_prev, C_prev, C_curr = C, C, C
        self.cells = nn.ModuleList()
        reduction_prev = True

        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)

        self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, x):
        logits_aux = None
        s0 = self.stem0(x)
        s1 = self.stem1(s0)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)

        # out = self.global_pooling(s1)
        # logits = self.classifier(out.view(out.size(0), -1))
        logits=self.head(s1)
        return logits
        # return logits, logits_aux
