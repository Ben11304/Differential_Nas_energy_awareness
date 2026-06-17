"""eval_genotype_dfuc v0.1.0 — re-evaluate a DART genotype on the §0k test split.

Fixes INTEGRITY flag M2: turns search-time `best_val_acc` (random 50/50 search
split, NOT comparable) into a TEST accuracy on the SAME block+dhash leakage-
controlled split the Swin-T teacher (§0k) used — so DART points sit directly next
to Swin RGB 0.797 / RGBD 0.817.

How comparability is guaranteed: the split + dataset + eval + parquet dump are
IMPORTED from dfuc2021_4class_benchmark.py (the exact §0k pipeline), not
re-implemented. Only the model differs: a fixed NetworkCIFAR built from the
discovered genotype, trained from scratch (no ImageNet init — DART nets are
from-scratch by construction).

RGB-only (input_channels=3) to match the Swin RGB anchor. seed splits the data
(--split_seed, mirror §0k) AND seeds the from-scratch training (--train_seed).

Run: python -u eval_genotype_dfuc.py --genotype_json <path> --out_dir <dir> ...
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# --- DART side: fixed-arch network + genotype type ---
from model import NetworkCIFAR
from genotypes import Genotype  # noqa: F401  (used by eval() of the genotype str)

# --- §0k side: import the EXACT split/dataset/eval/dump (do not re-implement) ---
BENCH_DIR = "/users/PGS0407/binben14/VietHuy/Foot-diabetic-depth-integrate/hybrid_cnn_vit"
if BENCH_DIR not in sys.path:
    sys.path.insert(0, BENCH_DIR)
import dfuc2021_4class_benchmark as bench  # noqa: E402


class GenoNet(nn.Module):
    """Fixed genotype net; returns logits only (drops NetworkCIFAR aux head)."""

    def __init__(self, genotype, C, layers, num_classes):
        super().__init__()
        self.net = NetworkCIFAR(C, num_classes, layers, auxiliary=False,
                                genotype=genotype, input_channels=3, device="cpu")

    def forward(self, x):
        logits, _ = self.net(x)
        return logits


def load_genotype(path):
    d = json.load(open(path))
    geno = eval(d["genotype"], {"Genotype": Genotype, "range": range})
    return geno, d


def build_0k_split(split_seed, block_size=10, dhash_threshold=5):
    """Reproduce the §0k block+dhash split exactly (same fns as the Swin teacher)."""
    paths, labels = bench.load_dfuc2021_paths()
    group_ids = bench.compute_dhash_groups(paths, threshold=dhash_threshold)
    (trp, trl), (vap, val), (tep, tel) = bench.block_stratified_split(
        paths, labels, block_size=block_size, seed=split_seed, group_ids=group_ids)
    return (trp, trl), (vap, val), (tep, tel)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--genotype_json", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--run_name", required=True)
    p.add_argument("--split_seed", type=int, default=42, help="§0k split seed")
    p.add_argument("--train_seed", type=int, default=42, help="from-scratch init seed")
    p.add_argument("--C", type=int, default=16)
    p.add_argument("--layers", type=int, default=5)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.025)
    p.add_argument("--image_size", type=int, default=96)
    p.add_argument("--label_smooth", type=float, default=0.05)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--save_ckpt", action="store_true",
                   help="Write best.pth (weights + genotype/build meta) for ONNX export.")
    args = p.parse_args()

    torch.manual_seed(args.train_seed)
    np.random.seed(args.train_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    geno, src = load_genotype(args.genotype_json)
    print(f"genotype source: {args.genotype_json} (search seed={src.get('seed')}, "
          f"ew={src.get('energy_weight')}, variant={src.get('variant')})", flush=True)

    (trp, trl), (vap, val), (tep, tel) = build_0k_split(args.split_seed)
    print(f"§0k split (seed={args.split_seed}): train={len(trp)} val={len(vap)} "
          f"test={len(tep)}", flush=True)
    print(f"test class dist: {dict(Counter(tel))}", flush=True)

    # bench RGBDataset honours §0k augmentation; image_size handled by its tf.
    train_ds = bench.RGBDataset(trp, trl, training=True, strong=True)
    test_ds = bench.RGBDataset(tep, tel, training=False, strong=False)
    val_ds = bench.RGBDataset(vap, val, training=False, strong=False)
    tl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=4, pin_memory=True, drop_last=True)
    vl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    tstl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = GenoNet(geno, args.C, args.layers, len(bench.DFUC2021_CLASSES)).to(device)
    n_params = sum(x.numel() for x in model.parameters())
    print(f"GenoNet params: {n_params/1e6:.3f}M  (C={args.C} layers={args.layers})", flush=True)

    cw = bench.class_weights_inv_freq(trl, len(bench.DFUC2021_CLASSES)).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=3e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    epochs = 1 if args.smoke else args.epochs
    best_val_f1, best_state = -1.0, None
    history = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        run = 0.0
        for bi, (img, lbl) in enumerate(tl):
            if args.smoke and bi >= 3:
                break
            img, lbl = img.to(device), lbl.to(device)
            opt.zero_grad()
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    loss = F.cross_entropy(model(img), lbl, weight=cw,
                                           label_smoothing=args.label_smooth)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss = F.cross_entropy(model(img), lbl, weight=cw,
                                       label_smoothing=args.label_smooth)
                loss.backward(); opt.step()
            run += loss.item()
        sched.step()
        va, vf1, _, _ = bench.evaluate(model, vl, device, "rgb")
        history.append({"epoch": ep, "train_loss": run / max(bi, 1),
                        "val_acc": va, "val_f1": vf1})
        if vf1 > best_val_f1:
            best_val_f1 = vf1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if args.smoke or ep % 10 == 0 or ep == epochs:
            print(f"  ep {ep}: loss={run/max(bi,1):.4f} val_acc={va:.4f} val_f1={vf1:.4f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    ta, tf1, ypred, ytrue = bench.evaluate(model, tstl, device, "rgb")
    pcr = bench.per_class_report(ytrue, ypred, bench.DFUC2021_CLASSES)
    split_tag = f"block_dhash_test(bs=10,dhash=5,split_seed={args.split_seed})"

    per_seed = {
        "run_name": args.run_name,
        "genotype_source": str(args.genotype_json),
        "search_seed": src.get("seed"), "energy_weight": src.get("energy_weight"),
        "variant": src.get("variant"),
        "split": split_tag, "split_seed": args.split_seed, "train_seed": args.train_seed,
        "eval_protocol": "0k_block_dhash_test",  # NOT search-val random 50/50
        "retrain": {"C": args.C, "layers": args.layers, "epochs": epochs,
                    "image_size": args.image_size, "from_scratch": True},
        "params_M": n_params / 1e6,
        "test_acc": ta, "test_macro_f1": tf1, "best_val_f1": best_val_f1,
        "per_class_f1": {c: pcr[c]["f1"] for c in bench.DFUC2021_CLASSES},
        "n_train": len(trp), "n_val": len(vap), "n_test": len(tep),
        "elapsed_sec": time.time() - t0, "paper_ready": False,
    }
    (out_dir / "per_seed_metrics.json").write_text(json.dumps(per_seed, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    if args.save_ckpt:
        # best-val-F1 weights + the genotype/build args needed to reconstruct the
        # net for ONNX export (ENERGY hand-off). model already holds best_state.
        torch.save({"model_state": model.state_dict(),
                    "genotype_str": src["genotype"],
                    "C": args.C, "layers": args.layers,
                    "num_classes": len(bench.DFUC2021_CLASSES),
                    "input_channels": 3, "image_size": args.image_size,
                    "test_acc": ta, "test_macro_f1": tf1,
                    "split": split_tag, "train_seed": args.train_seed},
                   out_dir / "best.pth")
    bench.dump_predictions_parquet(model, test_ds, tstl, device, "rgb",
                                   args.train_seed, split_tag, out_dir)
    print(f"===> {args.run_name} TEST acc={ta:.4f} macro_f1={tf1:.4f} "
          f"(split={split_tag})", flush=True)


if __name__ == "__main__":
    main()
