import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from genotypes import PRIMITIVES
from genotypes import Genotype


energy_dict = {
    'none': [],
    'max_pool_3x3': [],
    'skip_connect': [],
    'sep_conv_3x3': [],
    'sep_conv_5x5': [],
    'dil_conv_3x3': [],
    'dil_conv_5x5': [],
    'conv_1x1': [],
     'grouped_conv': [],
    'alt_sep_conv': [],
     # 'batchnorm',
    'conv_3x3': [],
}


class SNNHead(nn.Module):
    """
    SNN head sử dụng multi-step spiking, gồm Conv2d (1x1) và Upsampling.
    Đầu ra trung bình spike sau num_steps để ổn định kết quả.
    """
    def __init__(self, C_prev, num_classes, num_steps=10, beta=0.95, scale_factor=4):
        super(SNNHead, self).__init__()
        self.num_steps = num_steps
        self.scale_factor = scale_factor
        
        # Convolution 1x1 chuyển channels về num_classes
        self.conv = nn.Conv2d(C_prev, num_classes, kernel_size=1, bias=False)
        
        # Neuron Leaky Integrate-and-Fire
        self.lif = snn.Leaky(beta=beta, learn_beta=True)
        
        # Upsample để khôi phục kích thước ban đầu
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)

    def forward(self, x):
        mem = self.lif.init_leaky()
        spk_acc = 0
        
        for _ in range(self.num_steps):
            cur = self.conv(x)
            spk, mem = self.lif(cur, mem)
            spk_acc += spk
        
        # Trung bình spike qua các bước thời gian
        out = spk_acc / self.num_steps
        
        # Upsample
        out = self.upsample(out)
        return out

class ConvHead(nn.Module):
    """
    Head dùng Conv2D bình thường để thay thế SNNHead.
    Gồm Conv 1x1 + Upsampling.
    Tính MACs cho phần conv 1x1.
    """
    def __init__(self, C_prev, num_classes, scale_factor=4):
        super(ConvHead, self).__init__()
        self.conv = nn.Conv2d(C_prev, num_classes, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(num_classes)
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        self.MACs = 0  # sẽ được cập nhật trong forward

    def forward(self, x):
        B, C_in, H, W = x.shape
        out = self.conv(x)
        out = self.bn(out)
        out = self.upsample(out)

        # Tính MACs cho conv1x1: C_in × C_out × H × W
        self.MACs = C_in * out.shape[1] * H * W
        return out


class MixedOp(nn.Module):
    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self._energy_flags = []  # cờ đã tính energy chưa
        self._macs = []  # cache energy sau forward

        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            self._ops.append(op)
            self._macs.append(torch.tensor(0.0, dtype=torch.float32))  # khởi đầu 0
            self._energy_flags.append(False)

    def forward(self, x, weights):
        total_macs=sum(w * op.energy for w, op in zip(weights, self._ops))        
        result=sum(w * op(x) for w, op in zip(weights, self._ops))
        return result, total_macs

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






class Cell(nn.Module):
    def __init__(self, steps, multiplier,
                 C_prev_prev, C_prev, C,
                 reduction, reduction_prev):
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
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        # print(f"s0 :{s0} and s1: {s1}")
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
        for i, s in enumerate(states[-self._multiplier:]):
            if not isinstance(s, torch.Tensor):
                print(f"[Error] states[{i}] is not a tensor! It is: {type(s)}")


        return torch.cat(states[-self._multiplier:], dim=1), total_cell_macs





class Stem(nn.Module):
    """
    Lớp Stem để xử lý đầu vào.
    Bao gồm Conv2d 3x3 và BatchNorm2d.
    Tính MACs cho conv đầu vào.
    """
    def __init__(self, C_out, C_in=4):
        super(Stem, self).__init__()
        self.conv = nn.Conv2d(C_in, C_out, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(C_out)
        self.MACs = 0  # sẽ cập nhật trong forward

    def forward(self, x):
        B, _, H, W = x.shape
        out = self.conv(x)
        out = self.bn(out)

        # MACs = C_in × C_out × K × K × H × W
        self.MACs = self.conv.in_channels * self.conv.out_channels * \
                    self.conv.kernel_size[0] * self.conv.kernel_size[1] * H * W
        return out



class Network(nn.Module):
    def __init__(self, C, num_classes, layers, criterion,
                 steps=4, multiplier=4, stem_multiplier=3,
                 device='cuda',snn_step=5):
        """
        Thêm tham số `device` để chỉ định chạy trên CPU hay GPU.
        Mặc định 'cuda' = GPU (nếu khả dụng), bạn cũng có thể dùng 'cpu'.
        """
        super(Network, self).__init__()
        self._C = C
        self._snn_step = snn_step
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self._device = device   # Lưu lại device

        C_curr = stem_multiplier * C
        # self.stem = nn.Sequential(
        #     nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
        #     nn.BatchNorm2d(C_curr)
        # )
        self.stem =Stem(C_out=C_curr)
        # self.stem=nn.Sequential(
        #     nn.Conv2d(4, C_curr, 3, padding=1, bias=False),  # đổi 3 thành 4
        #     nn.BatchNorm2d(C_curr)
        # )


        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False

        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(
                steps, multiplier,
                C_prev_prev, C_prev, C_curr,
                reduction, reduction_prev
            )
            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.head = ClassificationHead(C_prev=C_prev, num_classes=num_classes).to(self._device)


        # Khởi tạo alpha
        self._initialize_alphas()

        # Optionally, di chuyển chính mô hình về device
        self.to(self._device)

    def new(self):
        """
        Tạo một model mới nhưng sao chép giá trị alpha từ model gốc
        """
        model_new = Network(
            self._C, self._num_classes, self._layers, self._criterion,
            self._steps, self._multiplier, device=self._device
        )
        # Di chuyển về device
        model_new.to(self._device)

        # Copy dữ liệu alpha
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
            
        for Cell_new in model_new.cells:
            for MixedOp_new in Cell_new._ops:
                for op_new in MixedOp_new._ops:
                    # print(f"đây là operation { op_new } \n và flag của nó là {op_new.energy_flag} ")
                    op_new.energy_flag=1
                    
        # for Cell, Cell_new in self.cells,model_new.cells:
        #     for MixedOp,MixedOp_new in Cell._ops,Cell_new._ops:
        #         for op, op_new in MixedOp._ops, MixedOp_new._ops:
        #             op_new.flag=op.flag
                    
        return model_new

    # def forward(self, input):
    #     # Đảm bảo input cùng device
    #     # (Nếu bên ngoài ta đã input = input.to(device), có thể không cần dòng này)
    #     input = input.to(self._device)
    #     s1 = self.stem(input)
    #     s0 = s1
    #     for i, cell in enumerate(self.cells):
    #         if cell.reduction:
    #             weights = F.softmax(self.alphas_reduce, dim=-1)
    #         else:
    #             weights = F.softmax(self.alphas_normal, dim=-1)
    #         s0, s1 = s1, cell(s0, s1, weights)
    #     # out = self.global_pooling(s1)
    #     # logits = self.classifier(out.view(out.size(0), -1))
    #     logits=self.head(s1)
    #     return logits


    def forward(self, input):
        input = input.to(self._device)
        s1 = self.stem(input)
        # total_macs = self.stem.MACs 
        total_macs=0# ✅ Bắt đầu tính MACs từ stem
        s0 = s1
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = F.softmax(self.alphas_reduce, dim=-1)
            else:
                weights = F.softmax(self.alphas_normal, dim=-1)
    
            # Duyệt qua các MixedOp trong cell để cộng MACs (nếu có)
            offset = 0
            macs_cell = 0
            # for step in range(self._steps):
            #     for j in range(2 + step):
            #         op = cell._ops[offset + j]
            #         # Chỉ lấy MACs nếu có thuộc tính đó
            #         if hasattr(op, 'MACs'):
            #             macs_cell += op.MACs
            #     offset += 2 + step
            # total_macs += macs_cell
            
            s0, s1_mixed = s1, cell(s0, s1, weights)
            s1 = s1_mixed[0]
            mac_cells =s1_mixed[1]
            # print(f"total {mac_cells}")
            total_macs += s1_mixed[1]
        logits = self.head(s1)
        # total_macs += self.head.MACs  # ✅ MACs từ head
        # print("📊 Average Energy Consumption per Operation:")
        # for op, values in energy_dict.items():
        #     if len(values) > 0:
        #         avg_energy = sum(values) / len(values)
        #         print(f"🔋 {op:<15}: {avg_energy:.6f}")
        #     else:
        #         print(f"🔋 {op:<15}: No data")
        return logits, total_macs


    def _loss(self, input, target):
        input = input.to(self._device)
        target = target.to(self._device)
        logits,_ = self(input)
        return self._criterion(logits, target)
    # def _loss(self, input, target):
    #     input = input.to(self._device)
    #     target = target.to(self._device)
    #     logits, macs = self(input)
    #     return self._criterion(logits, target), macs


    def _initialize_alphas(self):
        # k là tổng số edge
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        # Khởi tạo alpha trên đúng device
        self.alphas_normal = nn.Parameter(
            1e-3 * torch.randn(k, num_ops, device=self._device)
        )
        self.alphas_reduce = nn.Parameter(
            1e-3 * torch.randn(k, num_ops, device=self._device)
        )

        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):
        """
        Trích xuất genotype từ alpha (áp dụng softmax) để biểu diễn kiến trúc.
        """
        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                # Chọn 2 edges lớn nhất bỏ qua 'none'
                edges = sorted(
                    range(i + 2),
                    key=lambda x: -max(
                        W[x][k] for k in range(len(W[x]))
                        if k != PRIMITIVES.index('none')
                    )
                )[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        # Lấy weights đã qua softmax
        # .detach() hoặc .data, .cpu().numpy() -> tách khỏi graph
        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).detach().cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).detach().cpu().numpy())

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype
