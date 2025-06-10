import torch
import torch.nn as nn
import pandas as pd
from ptflops import get_model_complexity_info
from dl_energy_estimator.energy_estimate import calculate_energy
import math



OPS = {
  'none' : lambda C, stride, affine: Zero(stride),
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
    # Hàm được sử dụng để tính toán năng lượng tiêu thụ và MACs của một chuỗi các lớp.
    mac_list = []
    energy_list = []
    current_x = x.clone()
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
                predicted_energy = calculate_energy(sample_df)
                energy_list.append(predicted_energy)
            else:
                print(f"sample is none, layer {layer}")
                    
                # forward để cập nhật shape
            with torch.no_grad():
                current_x = layer(current_x)
    energy = sum(energy_list)
    MACs = sum(mac_list)
    return energy,MACs


def create_energy_sample(layer, input_tensor):
    # Tạo DataFrame mô tả thông tin sẵn sàng cho dự đoán năng lượng bằng mô hình linear.
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
            "macs": 0 
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
    # Hàm tính toán số lượng MACs cho một module cụ thể.

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
        macs, _ = get_model_complexity_info(
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
    if self.energy_flag == 0:
            self.energy,self.MACs=energy(self.op,x)
            self.energy_flag = 1
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
            self.MACs = sum(self.mac_list)

        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out

