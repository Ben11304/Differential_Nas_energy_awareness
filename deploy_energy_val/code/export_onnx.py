"""export_onnx v0.1.0 — DART genotype net (best.pth) -> ONNX + parity check.

ENERGY hand-off: produces a CPU-portable ONNX graph (same graph the Pi4 runs) for
on-device energy measurement. Verifies parity against PyTorch via ONNX Runtime CPU
so the exported graph is numerically faithful before it leaves MODELING.

Input fixed 1x3x96x96 (batch=1, the Pi inference shape). Run on login/CPU.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import torch.nn as nn

import operations
operations.calculate_energy = lambda df: 0.0  # isolate MACs path on import
from genotypes import Genotype  # noqa: F401
from model import NetworkCIFAR


class GenoNet(nn.Module):
    """Fixed genotype net; returns logits only (drops aux). Inlined here so ONNX
    export does NOT import eval_genotype_dfuc -> bench/timm (keeps export fast +
    self-contained for the deploy bundle)."""

    def __init__(self, genotype, C, layers, num_classes):
        super().__init__()
        self.net = NetworkCIFAR(C, num_classes, layers, auxiliary=False,
                                genotype=genotype, input_channels=3, device="cpu")

    def forward(self, x):
        logits, _ = self.net(x)
        return logits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="path to best.pth")
    p.add_argument("--onnx_out", required=True)
    p.add_argument("--opset", type=int, default=13)
    p.add_argument("--rtol", type=float, default=1e-3)
    p.add_argument("--atol", type=float, default=1e-4)
    args = p.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    geno = eval(ck["genotype_str"], {"Genotype": Genotype, "range": range})
    model = GenoNet(geno, ck["C"], ck["layers"], ck["num_classes"])
    model.load_state_dict(ck["model_state"])
    model.eval()
    n_params = sum(x.numel() for x in model.parameters())

    sz = ck.get("image_size", 96)
    dummy = torch.randn(1, ck.get("input_channels", 3), sz, sz)
    with torch.no_grad():
        torch_logits = model(dummy).numpy()

    Path(args.onnx_out).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, args.onnx_out,
        input_names=["input"], output_names=["logits"],
        opset_version=args.opset,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        do_constant_folding=True,
    )

    # structural check + op inventory
    import onnx
    m = onnx.load(args.onnx_out)
    onnx.checker.check_model(m)
    ops = sorted({n.op_type for n in m.graph.node})

    # parity via ONNX Runtime CPU
    import onnxruntime as ort
    sess = ort.InferenceSession(args.onnx_out, providers=["CPUExecutionProvider"])
    ort_logits = sess.run(["logits"], {"input": dummy.numpy()})[0]

    max_abs = float(np.max(np.abs(torch_logits - ort_logits)))
    parity = bool(np.allclose(torch_logits, ort_logits, rtol=args.rtol, atol=args.atol))
    size_mb = Path(args.onnx_out).stat().st_size / 1e6

    result = {
        "ckpt": args.ckpt, "onnx": args.onnx_out, "opset": args.opset,
        "params_M": n_params / 1e6, "onnx_size_MB": size_mb,
        "test_acc": ck.get("test_acc"), "test_macro_f1": ck.get("test_macro_f1"),
        "parity_pass": parity, "max_abs_logit_diff": max_abs,
        "op_types": ops,
        "argmax_match": int(np.argmax(torch_logits) == np.argmax(ort_logits)),
    }
    print(json.dumps(result))
    Path(args.onnx_out).with_suffix(".parity.json").write_text(json.dumps(result, indent=2))
    if not parity:
        raise SystemExit(f"PARITY FAIL: max_abs_diff={max_abs:.2e}")


if __name__ == "__main__":
    main()
