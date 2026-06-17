"""Build MANIFEST.json for the energy-val deploy bundle + Kaggle package.

Reads the 4 exported .parity.json (acc/params/parity/ops/size) and pairs each
with its checkpoint + onnx path. No fabricated numbers — all pulled from the
export artifacts.
"""
import json
from pathlib import Path

BASE = Path("/users/PGS0407/binben14/VietHuy/DART_ulcer/Differential_Nas_energy_awareness")
BUNDLE = BASE / "deploy_energy_val"
MODELS = ["hybrid_s42_ew0.00", "hybrid_s42_ew0.90", "simple_s42_ew0.00", "simple_s42_ew0.90"]
ROLE = {"hybrid_s42_ew0.00": "best_hybrid", "hybrid_s42_ew0.90": "most_compressed",
        "simple_s42_ew0.00": "simple_dense", "simple_s42_ew0.90": "simple_compressed"}

entries = []
for m in MODELS:
    par = json.load(open(BASE / "onnx_export" / f"{m}.parity.json"))
    variant, _, ew = m.split("_")  # hybrid, s42, ew0.00
    entries.append({
        "model_id": m,
        "role": ROLE[m],
        "variant": variant,
        "search_seed": 42,
        "energy_weight": float(ew.replace("ew", "")),
        "params_M": par["params_M"],
        "test_acc_0k": par["test_acc"],
        "test_macro_f1_0k": par.get("test_macro_f1"),
        "eval_split": "0k_block_dhash_test (split_seed=42)",
        "input_shape": [1, 3, 96, 96],
        "onnx": f"onnx_models/{m}.onnx",
        "onnx_size_MB": par["onnx_size_MB"],
        "checkpoint": f"checkpoints/{m}.best.pth",
        "parity_pass": par["parity_pass"],
        "max_abs_logit_diff": par["max_abs_logit_diff"],
        "argmax_match": par["argmax_match"],
        "onnx_op_types": par["op_types"],
        "genotype_source": f"genotypes/macs_{variant}_ew{ew.replace('ew','')}.json",
    })

manifest = {
    "bundle": "DART energy-validation (Pi4 1GB, ONNX Runtime CPU)",
    "producer": "MODELING",
    "consumer": "ENERGY (physical measurement) + EVAL (accuracy↔energy Pareto)",
    "weights_note": "retrain train_seed=42; test_acc_0k is THIS weight's §0k test acc",
    "models": entries,
}
(BUNDLE / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
print(f"MANIFEST.json: {len(entries)} models")
for e in entries:
    print(f"  {e['model_id']:22s} {e['params_M']:.3f}M acc={e['test_acc_0k']:.4f} "
          f"onnx={e['onnx_size_MB']:.2f}MB parity={e['parity_pass']}")
