# DART energy-validation deploy bundle

Artifacts for **on-device (Raspberry Pi 4, 1GB) energy measurement** of DART
compression genotypes. Flow: user's Mac → SSH Pi4 → ONNX Runtime CPU inference →
measure energy/latency. MODELING produces code + weights + ONNX; **ENERGY** runs
the physical measurement (this bundle does NOT contain energy numbers).

## Layout
```
deploy_energy_val/
  code/         genotype→net + ONNX export (model.py, operations.py, utils.py,
                genotypes.py, model_search*.py, architect.py, export_onnx.py,
                eval_genotype_dfuc.py)
  genotypes/    30 searched genotype JSONs (simple/hybrid × 3 seeds × 5 EW)
  onnx_models/  4 front-point ONNX graphs (1×3×96×96, opset13, dynamic batch)
  checkpoints/  4 best.pth (weights + genotype_str + build meta + test_acc)
  MANIFEST.json model_id ↔ test_acc ↔ params ↔ genotype source ↔ parity
```

## Models exported (front points, RGB 3×96×96, §0k test acc)
See `MANIFEST.json` for exact numbers. All 4 ONNX verified bit-faithful vs PyTorch
via ONNX Runtime CPU (parity rtol 1e-3).

## How ENERGY runs it on the Pi
1. `pip install onnxruntime numpy pillow` on the Pi (CPU wheel).
2. Load an `onnx_models/*.onnx` with `onnxruntime.InferenceSession(..., providers=["CPUExecutionProvider"])`.
3. Feed a 1×3×96×96 float32 tensor (ImageNet-normalized RGB, 96px). Output = logits[1×4].
4. Measure wall energy (USB power meter / INA219) over N inferences; pair with
   `test_acc` from MANIFEST for the accuracy↔energy Pareto point.

## Notes / provenance
- Weights are from a **retrain** (train_seed 42) of each genotype — test_acc in
  MANIFEST is that retrain's §0k block+dhash test acc (the weights you measure).
  May differ ~noise from the search-sweep eval array (cudnn nondeterministic).
- `measure_pi4.py` is **NOT included** — owned by ENERGY (not yet provided).
- Energy/MACs side-channel in `operations.py` is no-op'd during ONNX export
  (graph/weights unaffected); the Pi runs pure ONNX Runtime, no DART code.
- ONNX op set is plain conv/bn/relu/add/pool/gemm — portable to ARM CPU.
