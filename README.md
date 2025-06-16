
# 🧠 Differential NAS with Energy Awareness – User Guide

## 📖 Overview

This repository contains the implementation of a **Differential Neural Architecture Search (NAS)** framework with **energy awareness**. The goal is to optimize neural network architectures by considering both **accuracy** and **energy consumption** during the search and training process.

---

## 📁 Project Structure

- **`operation.py`**  
  Defines operations as a forwardable `nn.Module` class. Each operation includes:
  - `self.energy` (initially 0, updated after the first forward pass)
  - `self.energy_flag` (initially 0, set to 1 after energy calculation)

- **`model_search.py`**  
  Fully defines the NAS model, including:
  - Cell architecture, mixops, and alpha weights
  - `MixOps` for output and energy values
  - `new_model` function to reset weights while retaining the alpha set

- **`train_search.py`**  
  Manages the NAS training process:
  - Integrates the model from `model_search.py`
  - Defines primitive sets for the search space
  - Logs to `./log` and supports integration with **Weights & Biases (wandb)**

- **`architect.py`**  
  Implements the update algorithm for the alpha weights:
  - Normalizes accuracy and energy loss in `backward_step`
  - Applies weighted loss:  
    `total_loss = 0.9 * loss_task_normalized + 0.05 * energy_normalized`

- **`train.py`**  
  Trains the selected architecture **after NAS**, passing results into the genotype.

- **`model.py`**  
  Supports training based on the final architecture (genotype).

- **`dl_energy_estimator/`**  
  Contains a model to predict energy consumption of operations.

---

## 🛠️ How to Run

### ✅ Prerequisites

- Install dependencies 
- Unzip the `dl_energy_estimator` module

---

### ⚙️ Steps

#### 1. Configure NAS Search

Edit `train_search.py`:
- Declare appropriate **primitive sets**
- Tune energy and accuracy weights in `backward_step` (inside `architect.py`)

Run the NAS training:
```bash
python train_search.py --device cuda --learning_rate 0.01 --epochs 50
````

#### 2. Train Final Architecture

After NAS search, assign the **final genotype** in `train_cifar.py` (modified for CIFAR-10).

Run:

```bash
python train_cifar.py --batch_size 16 --epochs 30 --gpu 1 --learning_rate 0.01
```

---

## 📝 Notes

* To use a different dataset, modify `self.stem` and `self.head` in `model_search.py`
* Training progress is logged in the `./log` folder and tracked via **wandb**

---
