#!/usr/bin/env python3
"""measure_pi4.py — Energy-validation harness cho Raspberry Pi 4 (1GB).

Theo protocol: DFU-Pipeline-AGENT/ENERGY/outputs/protocol_pi4_1gb_paperB.md
Platform CHỐT: Pi4 1GB / ONNX Runtime CPU / standalone-cell / input 96x96 / fp32.

Power meter của user KHÔNG log CSV -> đọc tay. Script tách 2 phase:
  --mode run    : Phase 1 AUTO trên Pi (latency/throughput/memory/thermal +
                  in mốc "ĐỌC POWER BÂY GIỜ"). Dump <id>_pi4_auto.json.
  --mode merge  : Phase 2 đọc recording sheet (manual power) -> derive J/inf
                  -> JSON cuối khớp shared/handoff_schema.md §measurements.

Chỉ phụ thuộc: python3, onnxruntime, numpy (Phase 1). psutil OPTIONAL (fallback
đọc /proc). KHÔNG phụ thuộc lib HPC. vcgencmd OPTIONAL (chỉ có trên Pi).

Cách gọi — xem cuối file (__main__ docstring) hoặc INSTRUCTION_local_agent.md §4-5.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time

INPUT_SHAPE = (1, 3, 96, 96)   # CHỐT 96x96 (KHÔNG 224)
PROTOCOL_DOC = "ENERGY/outputs/protocol_pi4_1gb_paperB.md"
PLATFORM = "raspberry_pi4_1gb"


# ----------------------------------------------------------------------------- helpers
def _now():
    return time.time()


def _read_meminfo():
    """(mem_available_kb, swap_used_kb) từ /proc/meminfo. Không cần psutil."""
    info = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                k, _, rest = line.partition(":")
                info[k.strip()] = int(rest.strip().split()[0])  # kB
    except OSError:
        return None, None
    mem_avail = info.get("MemAvailable")
    swap_used = None
    if "SwapTotal" in info and "SwapFree" in info:
        swap_used = info["SwapTotal"] - info["SwapFree"]
    return mem_avail, swap_used


def _self_rss_kb():
    """RSS process hiện tại (kB). psutil nếu có, fallback /proc/self/status."""
    try:
        import psutil  # noqa
        return psutil.Process(os.getpid()).memory_info().rss // 1024
    except Exception:
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1])
        except OSError:
            pass
    return None


def _vcgencmd(arg):
    """Đọc vcgencmd (chỉ có trên Pi). None nếu không có."""
    try:
        out = subprocess.check_output(["vcgencmd"] + arg.split(),
                                      stderr=subprocess.DEVNULL, timeout=5)
        return out.decode().strip()
    except Exception:
        return None


def _temp_c():
    raw = _vcgencmd("measure_temp")   # "temp=47.2'C"
    if raw and "=" in raw:
        try:
            return float(raw.split("=")[1].split("'")[0])
        except (IndexError, ValueError):
            return None
    return None


def _throttled():
    raw = _vcgencmd("get_throttled")  # "throttled=0x0"
    if raw and "=" in raw:
        return raw.split("=")[1]
    return None


def _percentile(sorted_vals, p):
    """Percentile linear-interp; sorted_vals đã sort tăng dần."""
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _stats(vals):
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n if n > 1 else 0.0
    return {"mean": mean, "std": var ** 0.5, "n": n}


# ----------------------------------------------------------------------------- swap monitor
class SwapMonitor(threading.Thread):
    """Poll swap-used @2Hz. Set self.tripped nếu vượt baseline + abort_kb."""
    def __init__(self, abort_kb):
        super().__init__(daemon=True)
        self.abort_kb = abort_kb
        self.tripped = False
        self.trip_msg = None
        self.peak_rss_kb = 0
        self._stop = threading.Event()
        _, self.base_swap = _read_meminfo()
        self.base_swap = self.base_swap or 0

    def run(self):
        while not self._stop.is_set():
            _, swap_used = _read_meminfo()
            rss = _self_rss_kb()
            if rss:
                self.peak_rss_kb = max(self.peak_rss_kb, rss)
            if swap_used is not None and swap_used - self.base_swap > self.abort_kb:
                self.tripped = True
                self.trip_msg = ("swap touched: +%dkB (baseline %dkB)"
                                 % (swap_used - self.base_swap, self.base_swap))
                return
            self._stop.wait(0.5)

    def stop(self):
        self._stop.set()


# ----------------------------------------------------------------------------- input
def _load_inputs(image_dir, input_name, shape):
    """Trả list ndarray float32 [1,3,96,96]. Dùng ảnh nếu có, fallback dummy."""
    import numpy as np
    if image_dir and os.path.isdir(image_dir):
        try:
            from PIL import Image
        except ImportError:
            Image = None
        files = []
        if Image:
            for fn in sorted(os.listdir(image_dir)):
                if fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    files.append(os.path.join(image_dir, fn))
        if files:
            imgs = []
            for fp in files:
                im = Image.open(fp).convert("RGB").resize((shape[3], shape[2]))
                a = np.asarray(im, dtype="float32") / 255.0          # HWC
                a = a.transpose(2, 0, 1)[None, ...]                  # 1CHW
                imgs.append(np.ascontiguousarray(a))
            print("[input] %d ảnh thật từ %s" % (len(imgs), image_dir))
            return imgs
        print("[input] %s không có ảnh hợp lệ -> dummy" % image_dir)
    print("[input] dùng dummy zeros %s" % (tuple(shape),))
    return [np.zeros(shape, dtype="float32")]


# ----------------------------------------------------------------------------- Phase 1
def phase_run(args):
    import numpy as np  # noqa
    try:
        import onnxruntime as ort
    except ImportError:
        sys.exit("[FATAL] onnxruntime chưa cài. pip install onnxruntime")

    shape = [int(x) for x in args.input_shape.split(",")]
    if len(shape) != 4:
        sys.exit("[FATAL] --input_shape phải 4 số, vd 1,3,96,96")

    # Pre-flight RAM
    mem_avail, _ = _read_meminfo()
    if mem_avail is not None:
        print("[preflight] MemAvailable = %.0f MB" % (mem_avail / 1024))
        if mem_avail / 1024 < args.min_free_mb:
            sys.exit("[ABORT] RAM trống %.0fMB < %dMB. Kill background."
                     % (mem_avail / 1024, args.min_free_mb))

    if args.swapoff:
        print("[preflight] (giả định swap đã off — chạy `sudo dphys-swapfile "
              "swapoff` trước nếu chưa)")

    # ONNX session (thread config protocol §2)
    so = ort.SessionOptions()
    so.intra_op_num_threads = args.intra_op
    so.inter_op_num_threads = args.inter_op
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_cpu_mem_arena = args.mem_arena
    sess = ort.InferenceSession(args.onnx, sess_options=so,
                                providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    print("[onnx] input '%s' shape=%s | threads intra=%d inter=%d arena=%s"
          % (inp.name, inp.shape, args.intra_op, args.inter_op, args.mem_arena))

    inputs = _load_inputs(args.image_dir, inp.name, shape)
    feed_list = [{inp.name: x} for x in inputs]
    nf = len(feed_list)

    rss_after_init = _self_rss_kb()
    print("[mem] RSS sau init = %s MB"
          % ("%.0f" % (rss_after_init / 1024) if rss_after_init else "?"))

    swapmon = SwapMonitor(abort_kb=args.swap_abort_mb * 1024)
    swapmon.start()

    def _check_swap():
        if swapmon.tripped:
            swapmon.stop()
            out = _auto_payload(args, shape, inp, None, None, None, None,
                                swapmon, invalid=swapmon.trip_msg)
            _dump_auto(args, out)
            sys.exit("[INVALID] %s -> đã mark INVALID, KHÔNG dùng số này."
                     % swapmon.trip_msg)

    temp_start = _temp_c()
    throttled_start = _throttled()

    # ---- IDLE marker ----
    print("\n" + "=" * 64)
    print(">>> IDLE READY. Model đã load, CHƯA chạy inference.")
    print(">>> ĐỌC 'idle_W' trên power meter BÂY GIỜ (chờ số ổn định ~%ds)."
          % args.idle_s)
    print(">>> [pp.B] ZERO cumulative counter (mWh/mAh) BÂY GIỜ nếu dùng pp.B.")
    print("=" * 64)
    if args.interactive_meter:
        input(">>> Đọc xong, ghi sheet, nhấn ENTER để warmup... ")
    else:
        time.sleep(args.idle_s)
    _check_swap()

    # ---- warmup ----
    print("[warmup] %d iter..." % args.warmup)
    for i in range(args.warmup):
        sess.run(None, feed_list[i % nf])
    _check_swap()

    # ---- RUN steady-state marker ----
    print("\n" + "=" * 64)
    print(">>> RUN bắt đầu. Loop tối thiểu %ds / %d repeat."
          % (args.window_s, args.repeats))
    print(">>> Khi watt trên meter ỔN ĐỊNH (~30-60s nữa) = STEADY-STATE.")
    print(">>> ĐỌC 'active_W' (pp.A) tại steady-state mỗi repeat.")
    print("=" * 64 + "\n")
    if args.interactive_meter:
        input(">>> Sẵn sàng đọc active. ENTER để chạy... ")

    all_lat_ms = []
    repeat_records = []
    total_inf = 0
    t_run_start = _now()

    for r in range(1, args.repeats + 1):
        rep_lat = []
        rep_start = _now()
        it = 0
        # loop tới khi đủ window_s VÀ đủ measure_iters
        while (_now() - rep_start) < args.window_s or it < args.measure_iters:
            t0 = time.perf_counter()
            sess.run(None, feed_list[it % nf])
            rep_lat.append((time.perf_counter() - t0) * 1000.0)
            it += 1
            if it % 200 == 0:
                _check_swap()
        rep_dur = _now() - rep_start
        all_lat_ms.extend(rep_lat)
        total_inf += it
        t_now = _temp_c()
        thr_now = _throttled()
        repeat_records.append({
            "repeat": r, "n_inferences": it, "duration_s": round(rep_dur, 2),
            "temp_C": t_now, "throttled": thr_now,
        })
        print(">>> REPEAT %d DONE: n_inf=%d dur=%.1fs temp=%s throttled=%s"
              % (r, it, rep_dur, t_now, thr_now))
        print(">>>   --> ĐỌC active_W (pp.A) / ΔmWh sau zero (pp.B) cho repeat %d"
              % r)
        if args.interactive_meter and r < args.repeats:
            input(">>>   Ghi sheet xong, ENTER cho repeat kế... ")
        _check_swap()

    total_dur = _now() - t_run_start
    temp_end = _temp_c()
    throttled_end = _throttled()
    swapmon.stop()

    print("\n" + "=" * 64)
    print(">>> DONE. total_inf=%d total_dur=%.1fs" % (total_inf, total_dur))
    print(">>> Điền recording sheet đầy đủ rồi chạy --mode merge.")
    print("=" * 64)

    out = _auto_payload(args, shape, inp, all_lat_ms, repeat_records,
                        (temp_start, temp_end), (throttled_start, throttled_end),
                        swapmon, total_inf=total_inf, total_dur=total_dur,
                        rss_after_init=rss_after_init, mem_avail=mem_avail)
    _dump_auto(args, out)


def _auto_payload(args, shape, inp, all_lat_ms, repeat_records, temps,
                  throttles, swapmon, total_inf=0, total_dur=0.0,
                  rss_after_init=None, mem_avail=None, invalid=None):
    lat_sorted = sorted(all_lat_ms) if all_lat_ms else []
    lat_block = _stats(all_lat_ms or [])
    if lat_sorted:
        lat_block.update({
            "p50": _percentile(lat_sorted, 50),
            "p95": _percentile(lat_sorted, 95),
            "p99": _percentile(lat_sorted, 99),
        })
    inf_per_s = (total_inf / total_dur) if total_dur else None
    payload = {
        "student_id": args.student_id,
        "platform": PLATFORM,
        "phase": "auto",
        "hardware": {
            "cpu": "Cortex-A72 4-core @1.5GHz",
            "ram_gb": 1, "gpu": None,
            "os": _os_string(),
            "power_meter": "[VERIFY model] MANUAL READ (no CSV)",
        },
        "protocol": {
            "runtime": _ort_version(),
            "intra_op_threads": args.intra_op, "inter_op_threads": args.inter_op,
            "cpu_mem_arena": args.mem_arena,
            "warmup_iters": args.warmup, "measure_iters": args.measure_iters,
            "repeats": args.repeats, "window_s": args.window_s,
            "batch_size": shape[0], "input_shape": shape, "fp_precision": "fp32",
            "swapoff": args.swapoff,
            "protocol_doc": PROTOCOL_DOC,
        },
        "latency_ms": lat_block,
        "throughput": {"inf_per_s": inf_per_s, "samples_per_joule": None},
        "n_inferences_total": total_inf,
        "duration_s_total": round(total_dur, 2),
        "per_repeat": repeat_records or [],
        "memory": {
            "peak_rss_mb": round(swapmon.peak_rss_kb / 1024, 1) if swapmon.peak_rss_kb else None,
            "rss_after_init_mb": round(rss_after_init / 1024, 1) if rss_after_init else None,
            "swap_touched": bool(swapmon.tripped),
            "mem_available_pre_mb": round(mem_avail / 1024, 1) if mem_avail else None,
        },
        "thermal": {
            "temp_start_c": temps[0] if temps else None,
            "temp_end_c": temps[1] if temps else None,
            "throttled_start": throttles[0] if throttles else None,
            "throttled_end": throttles[1] if throttles else None,
        },
        "power_pending_manual": True,
        "ts_utc": _utc(),
    }
    if invalid:
        payload["INVALID"] = invalid
        payload["power_pending_manual"] = False
    return payload


def _dump_auto(args, payload):
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "%s_pi4_auto.json" % args.student_id)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print("[write] %s" % path)
    # raw latency CSV (gitignored ở repo)
    if payload.get("per_repeat"):
        raw = os.path.join(args.out_dir, "%s_pi4_raw.csv" % args.student_id)
        with open(raw, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["repeat", "n_inferences", "duration_s", "temp_C", "throttled"])
            for r in payload["per_repeat"]:
                w.writerow([r["repeat"], r["n_inferences"], r["duration_s"],
                            r["temp_C"], r["throttled"]])
        print("[write] %s" % raw)


def _os_string():
    try:
        with open("/etc/os-release") as f:
            for line in f:
                if line.startswith("PRETTY_NAME="):
                    return line.split("=", 1)[1].strip().strip('"')
    except OSError:
        pass
    return "[VERIFY os]"


def _ort_version():
    try:
        import onnxruntime as ort
        return "onnxruntime %s" % ort.__version__
    except Exception:
        return "onnxruntime [VERIFY]"


def _utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ----------------------------------------------------------------------------- Phase 2
def phase_merge(args):
    with open(args.auto) as f:
        auto = json.load(f)
    if auto.get("INVALID"):
        sys.exit("[ABORT] auto.json mark INVALID (%s). Đo lại." % auto["INVALID"])

    rows = _read_sheet(args.sheet, auto["student_id"])
    method = args.method.upper()

    net_list, gross_list, idle_list, active_list = [], [], [], []
    notes = []
    for row in rows:
        n_inf = _to_float(row.get("n_inferences"))
        dur = _to_float(row.get("duration_s"))
        idle_w = _to_float(row.get("idle_W"))
        if idle_w is not None:
            idle_list.append(idle_w)
        if method == "A":
            active_w = _to_float(row.get("active_W"))
            if None in (active_w, idle_w, n_inf, dur) or n_inf == 0:
                notes.append("repeat %s: thiếu field pp.A -> skip" % row.get("repeat"))
                continue
            active_list.append(active_w)
            gross = active_w * dur / n_inf
            net = (active_w - idle_w) * dur / n_inf
            gross_list.append(gross)
            net_list.append(net)
        elif method == "B":
            d_mwh = _to_float(row.get("delta_mWh"))
            if None in (d_mwh, n_inf) or n_inf == 0:
                notes.append("repeat %s: thiếu field pp.B -> skip" % row.get("repeat"))
                continue
            # ΔWh*3600/N = J/inf gross; idle: cần idle J/inf nếu đo riêng
            gross = (d_mwh / 1000.0) * 3600.0 / n_inf
            gross_list.append(gross)
            if idle_w is not None and dur:
                idle_j_per_inf = idle_w * dur / n_inf
                net_list.append(gross - idle_j_per_inf)
        else:
            sys.exit("[FATAL] --method phải A hoặc B")

    have_power = bool(gross_list)
    if have_power:
        energy = {
            "net": _stats(net_list) if net_list else "[NOT_MEASURED: net cần idle đo riêng]",
            "gross": _stats(gross_list),
        }
    else:
        energy = "[NOT_MEASURED: manual power not recorded]"
        notes.append("Không có power đọc tay hợp lệ -> energy NOT_MEASURED, "
                     "giữ latency/throughput.")

    # throughput samples/Joule từ gross mean (nếu có)
    tput = dict(auto.get("throughput", {}))
    if have_power and gross_list:
        gmean = sum(gross_list) / len(gross_list)
        tput["samples_per_joule"] = (1.0 / gmean) if gmean else None

    final = {
        "student_id": auto["student_id"],
        "platform": PLATFORM,
        "hardware": auto["hardware"],
        "protocol": dict(auto["protocol"], energy_method=("A_steady_state"
                         if method == "A" else "B_mWh_counter"),
                         power_source="manual_read"),
        "inference_energy_j": energy,
        "latency_ms": auto["latency_ms"],
        "power_w": {
            "idle": _stats(idle_list)["mean"] if idle_list else None,
            "active_mean": _stats(active_list)["mean"] if active_list else None,
            "active_std": _stats(active_list)["std"] if active_list else None,
            "_note": "manual read, W",
        },
        "throughput": tput,
        "uptime_s": {
            "sustained_throughput_samples_per_s": auto["throughput"].get("inf_per_s"),
            "duration_s": auto.get("duration_s_total"),
        },
        "memory": auto["memory"],
        "thermal": auto["thermal"],
        "manual_uncertainty": {
            "meter_resolution": args.meter_resolution,
            "est_rel_error_pct": args.rel_error_pct,
            "reads_per_mark": args.reads_per_mark,
            "claim_grade": "relative + order-of-magnitude, NOT precision",
        },
        "notes": "; ".join(notes) if notes else "manual merge OK (%d repeat power)"
                 % len(gross_list),
        "commit_hash": args.commit_hash or "[VERIFY]",
        "ts_utc": _utc(),
    }

    out = args.out or os.path.join(os.path.dirname(args.auto) or ".",
                                   "%s_pi4.json" % auto["student_id"])
    # append-only: nếu tồn tại -> _v2, _v3...
    out = _append_only_path(out)
    with open(out, "w") as f:
        json.dump(final, f, indent=2)
    print("[write] %s" % out)
    if not have_power:
        print("[WARN] energy = NOT_MEASURED (chưa điền power). Latency vẫn có.")


def _append_only_path(path):
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    v = 2
    while os.path.exists("%s_v%d%s" % (base, v, ext)):
        v += 1
    return "%s_v%d%s" % (base, v, ext)


def _read_sheet(sheet, student_id):
    rows = []
    with open(sheet) as f:
        reader = csv.DictReader(r for r in f if not r.lstrip().startswith("#"))
        for row in reader:
            mid = (row.get("model_id") or "").strip()
            if mid and mid.upper().startswith("EXAMPLE"):
                continue
            if mid and student_id and mid != student_id:
                continue
            rows.append(row)
    if not rows:
        sys.exit("[FATAL] sheet không có dòng dữ liệu cho %s" % student_id)
    return rows


def _to_float(v):
    if v is None:
        return None
    v = str(v).strip()
    if v == "" or v.startswith("<<"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


# ----------------------------------------------------------------------------- CLI
def main():
    p = argparse.ArgumentParser(
        description="Pi4 energy-validation harness (protocol_pi4_1gb_paperB.md)")
    sub = p.add_subparsers(dest="_legacy")  # giữ chỗ; dùng --mode
    p.add_argument("--mode", required=True, choices=["run", "merge"])
    p.add_argument("--student_id", required=True)
    # Phase 1
    p.add_argument("--onnx")
    p.add_argument("--image_dir", default=None,
                   help="thư mục ảnh 96x96; bỏ trống -> dummy zeros")
    p.add_argument("--input_shape", default="1,3,96,96")
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--measure_iters", type=int, default=200)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--window_s", type=float, default=240.0)
    p.add_argument("--idle_s", type=float, default=60.0)
    p.add_argument("--intra_op", type=int, default=4)
    p.add_argument("--inter_op", type=int, default=1)
    p.add_argument("--mem_arena", action="store_true", default=False)
    p.add_argument("--swapoff", action="store_true", default=False)
    p.add_argument("--swap_abort_mb", type=float, default=5.0)
    p.add_argument("--min_free_mb", type=float, default=400.0)
    p.add_argument("--interactive_meter", action="store_true", default=False)
    p.add_argument("--out_dir", default="shared/registries/measurements")
    # Phase 2
    p.add_argument("--auto", help="<id>_pi4_auto.json từ Phase 1")
    p.add_argument("--sheet", help="recording sheet CSV user điền")
    p.add_argument("--method", default="A", help="A=steady-state W | B=mWh counter")
    p.add_argument("--out", default=None)
    p.add_argument("--meter_resolution", default="[VERIFY] 0.01W / 1mWh")
    p.add_argument("--rel_error_pct", default="[VERIFY từ §4.1]")
    p.add_argument("--reads_per_mark", type=int, default=3)
    p.add_argument("--commit_hash", default=None)

    args = p.parse_args()
    if args.mode == "run":
        if not args.onnx:
            p.error("--mode run cần --onnx")
        phase_run(args)
    else:
        if not (args.auto and args.sheet):
            p.error("--mode merge cần --auto và --sheet")
        phase_merge(args)


if __name__ == "__main__":
    main()
