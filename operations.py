import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import pandas as pd
from ptflops import get_model_complexity_info
from dl_energy_estimator.energy_estimate import calculate_energy
import math
# import wandb

# wandb.init(project="nas_energy_monitor", name="energy_log_example")



OPS = {
  'none' : lambda C, stride, affine: Zero(stride),
  # 'avg_pool_3x3': lambda C, stride, affine: AvgPoolMAC(kernel_size=3, stride=stride, padding=1,C=C),
  'max_pool_3x3': lambda C, stride, affine: MaxPoolMAC(kernel_size=3, stride=stride, padding=1,C=C),
  'skip_connect': lambda C, stride, affine: SkipConnect(C, stride, affine=affine),
  'sep_conv_3x3' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
  'sep_conv_5x5' : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
  'sep_conv_7x7' : lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
  'dil_conv_3x3' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
  'dil_conv_5x5' : lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    'conv_1x1': lambda C, stride, affine: Conv1x1(C, stride, affine),
    'conv_3x3': lambda C, stride, affine: Conv3x3(C, stride, affine),

     'grouped_conv': lambda C, stride, affine: GroupedConv(C, C, groups=4, stride=stride, padding=1, affine=affine),
    'alt_sep_conv': lambda C, stride, affine: AltDepthwiseSeparableConv(C, kernel_size=3, stride=stride, padding=1, affine=affine),
    'batchnorm': lambda C, stride, affine: OnlyBN(C, affine),
    'relu': lambda C, stride, affine: OnlyReLU()


}

class OnlyReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.op = nn.ReLU(inplace=False)
        self.MACs = 0
        self.energy = 0
        self.energy_flag = 0

    def forward(self, x):
        return self.op(x)


class OnlyBN(nn.Module):
    def __init__(self, C, affine=True):
        super().__init__()
        self.op = nn.BatchNorm2d(C, affine=affine)
        self.MACs = 0
        self.energy = 0

    def forward(self, x):
        return self.op(x)


class Conv1x1(nn.Module):
    def __init__(self, C, stride=1, affine=True):
        super().__init__()
        self.op = nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, 1, stride=stride, padding=0, bias=False),
        nn.BatchNorm2d(C, affine=affine)
    )
        self.energy_flag = 0
        self.energy = 0
        self.MACs = 0

    def forward(self, x):
        if self.energy_flag == 0:
            self.energy, self.MACs = energy(self.op, x)
            self.energy_flag = 1
        return self.op(x)

class Conv3x3(nn.Module):
    def __init__(self, C, stride=1, affine=True):
        super().__init__()
        self.op = nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, 3, stride=stride, padding=1, bias=False),  # padding=1 => giữ nguyên HxW
        nn.BatchNorm2d(C, affine=affine)
    )
        self.energy_flag = 0
        self.energy = 0
        self.MACs = 0

    def forward(self, x):
        if self.energy_flag == 0:
            self.energy, self.MACs = energy(self.op, x)
            self.energy_flag = 1
        return self.op(x)

class AltDepthwiseSeparableConv(nn.Module):
    def __init__(self, C, kernel_size=3, stride=1, padding=1, affine=True):
        super().__init__()
        self.op = nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, kernel_size=3, stride=stride, padding=1, groups=C, bias=False),
        nn.BatchNorm2d(C, affine=affine)
    )
        self.energy_flag = 0
        self.energy = 0
        self.MACs = 0

    def forward(self, x):
        if self.energy_flag == 0:
            self.energy, self.MACs = energy(self.op, x)
            self.energy_flag = 1
        return self.op(x)



class GroupedConv(nn.Module):
    def __init__(self, C_in, C_out, groups=4, kernel_size=3, stride=1, padding=1, affine=True):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
            nn.ReLU(inplace=False)
        )
        self.energy_flag = 0
        self.energy = 0
        self.MACs = 0

    def forward(self, x):
        if self.energy_flag == 0:
            self.energy, self.MACs = energy(self.op, x)
            self.energy_flag = 1
        return self.op(x)



# sample = pd.DataFrame([{
#     "batch_size": 123,
#     "image_size": 127,
#     "kernel_size": 7,
#     "input_size":4,
#     "output_size":2,
#     "in_channels": 80,
#     "out_channels": 29,
#     "stride": 1,
#     "padding": 2,
#     "attributed": "conv2d",
#     "sub_attributed": "tanh",
#     "macs":2.18534484e+11
# }])



def energy(op,x):
    mac_list = []
    energy_list = []
    current_x = x.clone()  # giữ tensor đang được truyền
    for layer in op:
            if isinstance(layer, nn.BatchNorm2d):
                energy_list.append(0)
                mac_list.append(0)
                continue
            sample_df = create_energy_sample(layer, current_x)
            dummy_model = nn.Sequential(layer)
            mac = count_module_macs(dummy_model, current_x.shape)
            mac_list.append(mac)
            if sample_df is not None:
                attributed = sample_df["attributed"].iloc[0]
                sample_df["macs"]=mac
                    # if attributed=='conv2d':
                    #     print(sample_df.head(1))
                predicted_energy = calculate_energy(sample_df)
                energy_list.append(predicted_energy)
                    # if sample_df["sub_attributed"].iloc[0] != 'none':
                    #     sub=sample_df["sub_attributed"].iloc[0]
                        
                    #     print(sample_df.head(1)) 
                # print(f"📌 Dự đoán năng lượng tiêu thụ của {attributed} : {predicted_energy}")
            else:
                print(f"sample is none, layer {layer}")
                    
                # forward để cập nhật shape
            with torch.no_grad():
                current_x = layer(current_x)
                    
    energy = sum(energy_list)
    # print(f"total energy {energy}")
    MACs = sum(mac_list)
    # print(f"total macs {MACs}")
    return energy,MACs


def create_energy_sample(layer, input_tensor):
    sample = {"batch_size": input_tensor.shape[0]}
    C, H, W = input_tensor.shape[1:]

    if isinstance(layer, nn.Conv2d):
        sample.update({
            "attributed": "conv2d",
            "image_size": H,
            "kernel_size": layer.kernel_size[0] if isinstance(layer.kernel_size, tuple) else layer.kernel_size,
            "in_channels": layer.in_channels,
            "out_channels": layer.out_channels,
            "stride": layer.stride[0] if isinstance(layer.stride, tuple) else layer.stride,
            "padding": layer.padding[0] if isinstance(layer.padding, tuple) else layer.padding,
            "input_size": layer.in_channels,
            "output_size": layer.out_channels,
            "sub_attributed": "none",
            "macs": 0  # bạn cần tự điền thủ công MACs nếu không dùng ptflops
        })
    elif isinstance(layer, nn.ReLU):
        sample.update({
            "attributed": "activation",
            "sub_attributed": "relu",
            "input_size": C * H * W,
            "output_size": C * H * W,
            "macs": 0
        })
    elif isinstance(layer, nn.MaxPool2d):
        sample.update({
            "attributed": "maxpool2d",
            "image_size": H,
            "kernel_size": layer.kernel_size if isinstance(layer.kernel_size, int) else layer.kernel_size[0],
            "in_channels": C,
            "stride": layer.stride if isinstance(layer.stride, int) else layer.stride[0],
            "padding": layer.padding if isinstance(layer.padding, int) else layer.padding[0],
            "input_size": C * H * W,
            "output_size": C * H * W,
            "sub_attributed": "none",
            "macs": 0
        })
    else:
        return None

    return pd.DataFrame([sample])


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



class AvgPoolMAC(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, affine=False,C=1):
        super().__init__()
        self.avgpool = nn.AvgPool2d(kernel_size, stride=stride, padding=padding, count_include_pad=False)
        self.bn = nn.BatchNorm2d(C, affine=affine)  # batchnorm sau pool (giống logic wrap)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.MACs = 0

    def forward(self, x):
        B, C, H, W = x.shape
        out = self.avgpool(x)
        self.MACs = C * out.shape[2] * out.shape[3] * (self.kernel_size ** 2 - 1)
        return self.bn(out)




class SkipConnect(nn.Module):
    def __init__(self, C, stride, affine=True):
        super().__init__()
        if stride == 1:
            self.op = Identity()
            self.MACs = 5
        else:
            self.op = FactorizedReduce(C, C, affine)
            self.C = C
            self.MACs = 5  # Tính sau trong forward

        self.energy=0
        self.energy_flag=0

    def forward(self, x):
        out = self.op(x)
        if isinstance(self.op, FactorizedReduce):
            if self.energy_flag == 0:
                self.energy=self.op.energy
                self.MACs=self.op.MACs
                print(f" energy of skip connecter : {self.energy}")
                self.energy_flag = 1
        return out



class MaxPoolMAC(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, affine=False, C=1):
        super().__init__()
        self.op=nn.Sequential(
            nn.MaxPool2d(kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(C, affine=affine),
        )
        self.MACs = 5
        self.energy=0# MaxPool doesn't use MACs
        self.energy_flag=0
    def forward(self, x):
        if self.energy_flag == 0:
            self.energy,self.MACs=energy(self.op,x)
            self.energy_flag = 1
        return self.op(x)



class SNNDense(nn.Module):
    """
    Dense spiking neuron block (auto flatten + lazy input_dim inference).
    Có thể dùng cho cả [B, C, H, W] hoặc [B, D].
    """
    def __init__(self, output_dim, num_steps=5, beta=0.95):
        super(SNNDense, self).__init__()
        self.output_dim = output_dim
        self.num_steps = num_steps
        self.beta = beta

        self.fc = None  # lazy init
        self.lif = snn.Leaky(beta=self.beta, learn_beta=True)

    def forward(self, x):
        if x.ndim == 4:
            x = torch.flatten(x, start_dim=1)  # [B, C, H, W] -> [B, D]

        if self.fc is None:
            input_dim = x.size(1)
            self.fc = nn.Linear(input_dim, self.output_dim).to(x.device)

        mem = self.lif.init_leaky()
        spk_acc = 0

        for _ in range(self.num_steps):
            cur_input = self.fc(x)
            spk, mem = self.lif(cur_input, mem)
            spk_acc += spk

        return spk_acc / self.num_steps




class SNNMultiStepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, padding=1,
                 num_steps=5, beta=0.5, affine=True):
        super(SNNMultiStepConv, self).__init__()
        self.num_steps = num_steps
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Spiking neurons
        self.lif1 = snn.Leaky(beta=beta, threshold=0.9)
        self.lif2 = snn.Leaky(beta=beta, threshold=0.9)
        self.lif3 = snn.Leaky(beta=beta, threshold=0.9)

        # Convolution layers moved AFTER spiking
        self.depthwise = nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride,
                                   padding=padding, groups=C_in, bias=False)
        self.pointwise = nn.Conv2d(C_in, C_out, kernel_size=1, bias=False)
        self.final_conv = nn.Conv2d(C_out, C_out, kernel_size=1, bias=False)
        self.bn_out = nn.BatchNorm2d(C_out, affine=affine)

        self.MACs = 5  # MAC counter

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        B, _, H, W = x.shape
        spk_acc = 0
        total_spikes = 0

        for _ in range(self.num_steps):
            spk1, mem1 = self.lif1(x, mem1)
            spk2, mem2 = self.lif2(spk1, mem2)
            spk3, mem3 = self.lif3(spk2, mem3)

            # Only compute conv on spikes
            out = self.depthwise(spk3)
            out = self.pointwise(out)
            out = self.final_conv(out)
            out = self.bn_out(out)
            spk_acc += out

            total_spikes += spk3.detach().float().sum()

        avg_spike_rate = total_spikes / (B * self.C_in * H * W * self.num_steps)

        # MAC estimation: only computed when spikes occur
        expected_spikes = avg_spike_rate * B * self.C_in * H * W * self.num_steps
        dw_macs = expected_spikes * (self.kernel_size ** 2)
        pw_macs = expected_spikes * self.C_out
        final_macs = expected_spikes * self.C_out
        self.MACs = dw_macs + pw_macs + final_macs
        # print(f"[Optimized SNNMultiStepConv] 🔥 Avg Spike Rate: {avg_spike_rate:.4f}, MACs: {self.MACs:.2f}")

        return spk_acc / self.num_steps


class SpikingSepConv(nn.Module):
    def __init__(self, C, kernel_size=3, stride=1, padding=1, num_steps=3):
        super().__init__()
        self.num_steps = num_steps
        self.depthwise = nn.Conv2d(C, C, kernel_size, stride, padding, groups=C)
        self.pointwise = nn.Conv2d(C, C, 1)
        self.lif = snn.Leaky(beta=0.9)

    def forward(self, x):
        mem = self.lif.init_leaky()
        out_acc = 0
        for _ in range(self.num_steps):
            spk, mem = self.lif(x, mem)
            out = self.pointwise(self.depthwise(spk))
            out_acc += out
        return out_acc / self.num_steps



class SNN_IF_Conv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.if_neuron = snn.Leaky(beta=0.0)
        self.conv = nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        # spk = self.if_neuron(x)
        spk_tuple = self.if_neuron(x)
        if isinstance(spk_tuple, tuple):
            spk = spk_tuple[0]
        else:
            spk = spk_tuple
        out = self.conv(spk)

        out = self.conv(spk)
        return self.bn(out)


class SNNConv(nn.Module):
    """
    Ví dụ block spiking: [Spiking -> Conv(Depthwise) -> Conv(Pointwise) -> BN]
    thay thế ReLU bằng neuron snntorch (Leaky) cho single-step.
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SNNConv, self).__init__()
        # Khởi tạo neuron spiking, dùng spike_grad thay vì surrogate_function (tương thích phiên bản cũ hơn)
        self.lif = snn.Leaky(
            beta=0.9,           
            threshold=1.0,      
            spike_grad=surrogate.fast_sigmoid(), 
            learn_threshold=False
        )

        # Tương tự SepConv (Depthwise + Pointwise)
        self.depthwise = nn.Conv2d(
            C_in, C_in, kernel_size=kernel_size, 
            stride=stride, padding=padding, groups=C_in, bias=False
        )
        self.pointwise = nn.Conv2d(
            C_in, C_out, kernel_size=1, padding=0, bias=False
        )
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        # Single-step spiking: lif(x) có thể trả về:
        #   - 1 tensor (spk), hoặc
        #   - 1 tuple (spk, mem)
        spk = self.lif(x)
        
        # Nếu spk là tuple, chỉ lấy phần tử đầu (spike)
        if isinstance(spk, tuple):
            spk = spk[0]

        out = self.depthwise(spk)
        out = self.pointwise(out)
        out = self.bn(out)
        return out


class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)




class DilConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(DilConv, self).__init__()
    self.C_in = C_in
    self.C_out = C_out
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.dilation = dilation
    self.MACs = 5  # 📌 MAC counter
    self.op = nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation, groups=C_in, bias=False),
        nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
        nn.BatchNorm2d(C_out, affine=affine)
    )
    self.MACs = 0
    self.mac_list = [] 
    self.energy_list = [] # list of MACs for each sublayer
    self.energy_flag = 0
    self.energy=0

  def forward(self, x):
    out = self.op(x)
    # if self.energy_flag == 0:
    #     self.mac_list = []
    #     self.energy_list = []
    #     current_x = x.clone()  # giữ tensor đang được truyền
    #     for layer in self.op:
    #         sample_df = create_energy_sample(layer, current_x)
    #         dummy_model = nn.Sequential(layer)
    #         mac = count_module_macs(dummy_model, current_x.shape)
    #         self.mac_list.append(mac)
    #         if sample_df is not None:
    #             sample_df["macs"]=mac
    #             predicted_energy = calculate_energy(sample_df)
    #             self.energy_list.append(predicted_energy)
        
    #                 # forward để cập nhật shape
    #         with torch.no_grad():
    #             current_x = layer(current_x)
    # self.energy_flag = 1
    # self.energy = sum(self.energy_list)
    # # print(f"total energy {self.energy}")
    # self.MACs = sum(self.mac_list)
    if self.energy_flag == 0:
            self.energy,self.MACs=energy(self.op,x)
            self.energy_flag = 1
    # Tính MACs
    # H_out, W_out = out.shape[2], out.shape[3]
    # self.MACs = H_out * W_out * (self.C_in * self.kernel_size ** 2 + self.C_in * self.C_out)

    return out



class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),  # [0]
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=C_in, bias=False),           # [1]
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),   # [2]
            nn.BatchNorm2d(C_in, affine=affine),                           # [3]
            nn.ReLU(inplace=False),                                        # [4]
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1,
                      padding=padding, groups=C_in, bias=False),           # [5]
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),  # [6]
            nn.BatchNorm2d(C_out, affine=affine),                          # [7]
        )
        self.MACs = 0
        self.mac_list = [] 
        self.energy_list = [] # list of MACs for each sublayer
        self.energy_flag = 0
        self.energy=0

    def forward(self, x):
        if self.energy_flag == 0:
            self.energy,self.MACs=energy(self.op,x)
            self.energy_flag = 1
        # print(f" Inside energy value {self.energy}")
        return self.op(x)




class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


# class Zero(nn.Module):

#   def __init__(self, stride):
#     super(Zero, self).__init__()
#     self.stride = stride

#   def forward(self, x):
#     if self.stride == 1:
#       return x.mul(0.)
#     return x[:,:,::self.stride,::self.stride].mul(0.)

class Zero(nn.Module):
    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride
        self.MACs = 0
        self.energy = 0
        self.energy_flag=1

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)



class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)
        self.energy_flag = 0
        self.MACs = 0
        self.energy = 0
    def forward(self, x):
        if self.energy_flag == 0:
            self.mac_list = []
            self.energy_list = []
            sample_df = create_energy_sample(self.relu, x)
            mac = count_module_macs(self.relu, x.shape)
            self.mac_list.append(mac)
            if sample_df is not None:
                sample_df["macs"]=mac
                predicted_energy = calculate_energy(sample_df)
                self.energy_list.append(predicted_energy)
            x = self.relu(x)

            sample_df = create_energy_sample(self.conv_1, x)
            mac = count_module_macs(self.conv_1, x.shape)
            self.mac_list.append(mac)
            if sample_df is not None:
                sample_df["macs"]=mac
                predicted_energy = calculate_energy(sample_df)
                self.energy_list.append(predicted_energy)

            sample_df = create_energy_sample(self.conv_2, x[:, :, 1:, 1:])
            mac = count_module_macs(self.conv_2, x[:, :, 1:, 1:].shape)
            self.mac_list.append(mac)
            if sample_df is not None:
                sample_df["macs"]=mac
                predicted_energy = calculate_energy(sample_df)
                self.energy_list.append(predicted_energy)
            
            out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
            out = self.bn(out)
            self.energy_flag = 1
            self.energy = sum(self.energy_list)
            # print(f"total energy {self.energy}")
            self.MACs = sum(self.mac_list)

        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out

    # if self.energy_flag == 0:
    #     self.mac_list = []
    #     self.energy_list = []
    #     current_x = x.clone()  # giữ tensor đang được truyền
    #     for layer in self.op:
    #         sample_df = create_energy_sample(layer, current_x)
    #         dummy_model = nn.Sequential(layer)
    #         mac = count_module_macs(dummy_model, current_x.shape)
    #         self.mac_list.append(mac)
    #         if sample_df is not None:
    #             sample_df["macs"]=mac
    #             predicted_energy = calculate_energy(sample_df)
    #             self.energy_list.append(predicted_energy)
        
    #                 # forward để cập nhật shape
    #         with torch.no_grad():
    #             current_x = layer(current_x)
    # self.energy_flag = 1
    # self.energy = sum(self.energy_list)
    # # print(f"total energy {self.energy}")
    # self.MACs = sum(self.mac_list)






# class FactorizedReduce(nn.Module):
#     def __init__(self, C_in, C_out, affine=True):
#         super(FactorizedReduce, self).__init__()
#         assert C_out % 2 == 0

#         # ReLU (đặt riêng)
#         self.relu = nn.ReLU(inplace=False)

#         # Hai nhánh conv riêng biệt
#         self.path1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
#         self.path2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)

#         # BatchNorm sau khi concat
#         self.bn = nn.BatchNorm2d(C_out, affine=affine)

#         # Gộp tất cả vào 1 danh sách các layer để duyệt được
#         self.op = nn.ModuleList([
#             self.relu,
#             self.path1,
#             self.path2,
#             self.bn
#         ])

#         # Cho phép ghi nhận energy
#         self.energy = 0
#         self.MACs = 0
#         self.energy_flag = 0

#     def forward(self, x):
#         x_relu = self.relu(x)
#         out1 = self.path1(x_relu)
#         out2 = self.path2(x_relu[:, :, 1:, 1:])
#         out = torch.cat([out1, out2], dim=1)
#         out = self.bn(out)
#         return out

#     def __iter__(self):
#         # Cho phép for layer in model:
#         return iter(self.op)


