import torch
import time
import json
from operations import OPS
from codecarbon import EmissionsTracker

device = torch.device("cuda:1")
print(f"Using device: {device}")

C_list = [32]
HW_list = [32]
batch_size = 2
NUM_RUNS = 100



def compute_output_dim(H_in, W_in, kernel_size, stride, padding):
    H_out = (H_in + 2 * padding - kernel_size) // stride + 1
    W_out = (W_in + 2 * padding - kernel_size) // stride + 1
    return H_out, W_out

def compute_sepconv_macs(C, H, W, kernel_size, stride, padding):
    H_out, W_out = compute_output_dim(H, W, kernel_size, stride, padding)
    macs = H_out * W_out * (
        C * kernel_size * kernel_size +  # depthwise
        C * 1 * 1 +                      # pointwise
        C * kernel_size * kernel_size + # second depthwise
        C * C                           # second pointwise
    )
    return macs

def compute_dilconv_macs(C_in, C_out, H, W, kernel_size, stride, padding, dilation):
    H_out, W_out = compute_output_dim(H, W, kernel_size, stride, padding)
    return H_out * W_out * (C_in * kernel_size * kernel_size + C_in * C_out)

def compute_pool_macs(C, H, W, kernel_size, stride):
    H_out, W_out = compute_output_dim(H, W, kernel_size, stride, kernel_size//2)
    return H_out * W_out * C * kernel_size * kernel_size

def compute_snn_multistep_macs(C_in, C_out, H, W, kernel_size, stride, padding, num_steps):
    H_out, W_out = compute_output_dim(H, W, kernel_size, stride, padding)
    conv1x1_macs = H_out * W_out * C_out  # cuối
    macs_per_step = H_out * W_out * C_in * C_out * kernel_size * kernel_size + conv1x1_macs
    return macs_per_step * num_steps

def compute_if_macs(C_in, C_out, H, W, kernel_size, stride, padding):
    H_out, W_out = compute_output_dim(H, W, kernel_size, stride, padding)
    return H_out * W_out * C_in * C_out * kernel_size * kernel_size





PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'snn_multistep_3x3',
    'snn_multistep_5x5',
    # 'if_conv_3x3',
    # 'if_conv_5x5',
    # 'conv_7x1_1x7',
    # 'snn_dense',
]

energy_lookup = {}

for C in C_list:
    for H in HW_list:
        W = H
        input_tensor = torch.randn(batch_size, C, H, W).to(device)
        macs=0
        for op_name in PRIMITIVES:
            key = f"{op_name}_C{C}_H{H}_W{W}_bs{batch_size}"
            print(f"Benchmarking {key}")
            if "sep_conv_3x3" in op_name:
                macs = compute_sepconv_macs(C, H, W, 3, 1, 1)
            elif "sep_conv_5x5" in op_name:
                macs = compute_sepconv_macs(C, H, W, 5, 2, 2)
            elif "dil_conv_3x3" in op_name:
                macs = compute_dilconv_macs(C, C, H, W, 3, 1, 2, 2)
            elif "dil_conv_5x5" in op_name:
                macs = compute_dilconv_macs(C, C, H, W, 5, 1, 4, 2)
            elif "avg_pool" in op_name or "max_pool" in op_name:
                macs = compute_pool_macs(C, H, W, 3, 1)
            elif "snn_multistep_3x3" in op_name:
                macs = compute_snn_multistep_macs(C, C, H, W, 3, 1, 1, 3)
            elif "snn_multistep_5x5" in op_name:
                macs = compute_snn_multistep_macs(C, C, H, W, 5, 1, 2, 3)
            else:
                macs = None  # hoặc 0 nếu muốn mặc định

            op = OPS[op_name](C, 1, True).to(device)
            op.eval()

            # Warmup
            with torch.no_grad():
                for _ in range(20):
                    _ = op(input_tensor)

            tracker = EmissionsTracker(measure_power_secs=0.1, log_level="error")
            tracker.start()
            start = time.time()

            with torch.no_grad():
                for _ in range(NUM_RUNS):
                    _ = op(input_tensor)
            end = time.time()

            emissions = tracker.stop()
            print(f" operation {op_name} have: {op.MACs} mac")
            energy_kwh = tracker.final_emissions_data.energy_consumed
            total_time = end - start
            avg_time_per_batch = total_time / NUM_RUNS
            avg_time_per_sample = avg_time_per_batch / batch_size
            power_watt = (energy_kwh *1000 * 3600) / total_time
            Mac=int(op.MACs)

            energy_lookup[key] = {
                "avg_inference_time_per_batch": avg_time_per_batch,
                "avg_inference_time_per_sample": avg_time_per_sample,
                "emissions": emissions,
                "energy": energy_kwh,
                "power": power_watt,
                "macs": Mac,
                "total_time": total_time,
                "runs": NUM_RUNS,
                "batch_size": batch_size
            }

# Save
with open("energy_lookup_batch.json", "w") as f:
    json.dump(energy_lookup, f, indent=4)

print("Saved lookup table to energy_lookup_batchsize16.json")
