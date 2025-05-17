import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from codecarbon import EmissionsTracker
import snntorch as snn
from snntorch import surrogate

# ---------- CNN ----------
class SimpleCNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ---------- SNN ----------
class SimpleSNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=10, beta=0.9, num_steps=20):
        super(SimpleSNN, self).__init__()
        self.num_steps = num_steps

        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.lif1 = snn.Leaky(beta=beta)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.lif2 = snn.Leaky(beta=beta)
        self.pool2 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.lif3 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        spk_acc = 0
        total_spikes = 0
        total_neurons = 0

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        for _ in range(self.num_steps):
            cur = self.conv1(x)
            spk1, mem1 = self.lif1(cur, mem1)
            out = self.pool1(spk1)

            cur = self.conv2(out)
            spk2, mem2 = self.lif2(cur, mem2)
            out = self.pool2(spk2)

            out = out.view(out.size(0), -1)
            cur = self.fc1(out)
            spk3, mem3 = self.lif3(cur, mem3)

            out = self.fc2(spk3)
            spk_acc += out

            total_spikes += spk1.sum() + spk2.sum() + spk3.sum()
            total_neurons += spk1.numel() + spk2.numel() + spk3.numel()

        avg_spike_rate = total_spikes / (total_neurons * self.num_steps)
        return spk_acc / self.num_steps, avg_spike_rate.item()

# ---------- Energy Measure ----------
def measure_model_energy(model, input_tensor, name, is_snn=False):
    tracker = EmissionsTracker(measure_power_secs=0.1, log_level="error", output_file=f"{name}.csv")
    tracker.start()
    start = time.time()

    total_spike_rate = 0
    with torch.no_grad():
        for _ in range(1000):
            if is_snn:
                _, rate = model(input_tensor)
                total_spike_rate += rate
            else:
                _ = model(input_tensor)

    end = time.time()
    tracker.stop()
    energy_kwh = tracker.final_emissions_data.energy_consumed
    total_time = end - start
    power_watt = (energy_kwh * 1000 * 3600) / total_time

    print(f"\n🔋 Model: {name}")
    print(f"  ⚡ Power: {power_watt:.4f} W")
    print(f"  ⏱  Time: {total_time:.2f} s")
    print(f"  🔌 Energy: {energy_kwh:.6f} kWh")
    if is_snn:
        print(f"  🧠 Avg Spike Rate: {total_spike_rate / 1000:.4f}")

# ---------- Run ----------
device = torch.device("cuda:0")
dummy_input = torch.randn(16, 3, 32, 32).to(device)

cnn = SimpleCNN().to(device).eval()
snn = SimpleSNN().to(device).eval()

measure_model_energy(cnn, dummy_input, "cnn_model")
measure_model_energy(snn, dummy_input, "snn_model", is_snn=True)
