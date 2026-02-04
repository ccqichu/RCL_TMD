import argparse
import os
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class RunResult:
    name: str
    cmd: List[str]
    best_acc: Optional[float]
    cid_m_t_mean: Optional[float]
    cid_m_v_mean: Optional[float]
    log_path: str


def _parse_metrics(log_text: str):
    best_acc = None
    cid_m_t = None
    cid_m_v = None
    # Matches: "New best test_acc 0.876543 at epoch 3"
    for match in re.finditer(r"New best test_acc\s+([0-9.]+)", log_text):
        best_acc = float(match.group(1))
    # Matches: "Best model CID stats: m_t_mean=..., m_v_mean=..."
    match = re.search(r"Best model CID stats: m_t_mean=([0-9.]+), m_v_mean=([0-9.]+)", log_text)
    if match:
        cid_m_t = float(match.group(1))
        cid_m_v = float(match.group(2))
    return best_acc, cid_m_t, cid_m_v


def _run_one(name: str, cmd: List[str], logs_dir: str) -> RunResult:
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"{name}.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(" ".join(cmd) + "\n\n")
        f.flush()
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        proc.wait()

    with open(log_path, "r", encoding="utf-8") as f:
        log_text = f.read()
    best_acc, cid_m_t, cid_m_v = _parse_metrics(log_text)
    return RunResult(name, cmd, best_acc, cid_m_t, cid_m_v, log_path)

def _get_free_mem_mb() -> List[int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
        )
        return [int(x.strip()) for x in out.splitlines() if x.strip()]
    except Exception:
        return []


def _pick_available_gpus(gpus: List[int], min_free_mb: int, busy: set, allow_no_query: bool) -> List[int]:
    free_list = _get_free_mem_mb()
    if not free_list:
        if allow_no_query:
            return [g for g in gpus if g not in busy]
        return []
    available = []
    for g in gpus:
        if g in busy:
            continue
        if g < len(free_list) and free_list[g] >= min_free_mb:
            available.append(g)
    return available


def main():
    parser = argparse.ArgumentParser(description="Grid search for main metric vs CID visualization")
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--base_output_dir", default="/home/user/chengtaiyu/RCLMuFN-main_copy/output_dir/grid")
    parser.add_argument("--logs_dir", default="/home/user/chengtaiyu/RCLMuFN-main_copy/seed_runs/grid_logs")
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU ids to use")
    parser.add_argument("--min_free_mb", type=int, default=12000, help="Minimum free memory (MB) to launch a run")
    parser.add_argument("--poll_interval", type=int, default=20, help="Seconds between GPU/memory checks")
    parser.add_argument("--allow_no_query", action="store_true",
                        help="If nvidia-smi query fails, allow launching on any idle GPU")
    parser.add_argument("--sequential", action="store_true",
                        help="Run one job at a time, but pick any GPU with enough free memory")
    parser.add_argument("--per_gpu_queue", type=int, default=0,
                        help="If >0, assign this many jobs per GPU in round-robin, and run sequentially per GPU")
    args = parser.parse_args()

    # Define a compact grid. Adjust ranges as needed.
    grid = {
        "rho": [0.6, 0.7, 0.8],
        "rho_t": [0.75, 0.85, 0.9],
        "tau_min": [0.9, 1.2, 1.5],
        "cid_smooth_beta": [2.0, 2.5, 3.0],
        "lambda_end": [1e-4, 3e-4, 5e-4],
    }

    base_cmd = [
        "python",
        "/home/user/chengtaiyu/RCLMuFN-main_copy/src/main.py",
        "--num_train_epochs",
        str(args.epochs),
        "--lambda_ratio_start",
        "0",
        "--lambda_itm_start",
        "0",
        "--lambda_schedule",
        "none",
        "--tau_schedule_mode",
        "step",
        "--tau_decay",
        "0.99995",
        "--neg_sampling",
        "shuffle",
    ]

    results: List[RunResult] = []
    pending: List[Dict[str, str]] = []
    run_idx = 0
    for rho in grid["rho"]:
        for rho_t in grid["rho_t"]:
            for tau_min in grid["tau_min"]:
                for beta in grid["cid_smooth_beta"]:
                    for lam in grid["lambda_end"]:
                        run_idx += 1
                        name = f"run{run_idx}_rho{rho}_rhot{rho_t}_tau{tau_min}_b{beta}_lam{lam}"
                        out_dir = os.path.join(args.base_output_dir, name)
                        cmd = base_cmd + [
                            "--output_dir", out_dir,
                            "--lambda_ratio_end", str(lam),
                            "--lambda_itm_end", str(lam),
                            "--tau_min", str(tau_min),
                            "--rho", str(rho),
                            "--rho_t", str(rho_t),
                            "--cid_smooth_beta", str(beta),
                        ]
                        pending.append({"name": name, "cmd": cmd})

    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    running = []  # list of dicts: {proc, name, cmd, gpu, log_path}

    # If per_gpu_queue is set, keep up to N concurrent runs per GPU.
    if args.per_gpu_queue > 0:
        def _launch_on(gpu_id: int):
            available = _pick_available_gpus([gpu_id], args.min_free_mb, set(), args.allow_no_query)
            if not available:
                return False
            if not pending:
                return False
            job = pending.pop(0)
            name = job["name"]
            cmd = job["cmd"] + ["--device", str(gpu_id)]
            log_path = os.path.join(args.logs_dir, f"{name}.log")
            os.makedirs(args.logs_dir, exist_ok=True)
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(" ".join(cmd) + "\n\n")
                f.flush()
                print(f"Running {name} on GPU {gpu_id} ...")
                proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
            running.append({"proc": proc, "name": name, "cmd": cmd, "gpu": gpu_id, "log_path": log_path})
            return True

        # Launch up to per_gpu_queue per GPU to start
        for g in gpus:
            while sum(1 for item in running if item["gpu"] == g) < args.per_gpu_queue:
                if not _launch_on(g):
                    break

        while pending or running:
            still_running = []
            finished = []
            for item in running:
                if item["proc"].poll() is None:
                    still_running.append(item)
                else:
                    finished.append(item)
            running = still_running
            for item in finished:
                with open(item["log_path"], "r", encoding="utf-8") as f:
                    log_text = f.read()
                best_acc, cid_m_t, cid_m_v = _parse_metrics(log_text)
                results.append(RunResult(item["name"], item["cmd"], best_acc, cid_m_t, cid_m_v, item["log_path"]))
            running = still_running
            for g in gpus:
                while sum(1 for item in running if item["gpu"] == g) < args.per_gpu_queue:
                    if not _launch_on(g):
                        break
            time.sleep(args.poll_interval)
    else:
        while pending or running:
            # Check finished runs
            still_running = []
            for item in running:
                if item["proc"].poll() is None:
                    still_running.append(item)
                    continue
                # Finished: parse log
                with open(item["log_path"], "r", encoding="utf-8") as f:
                    log_text = f.read()
                best_acc, cid_m_t, cid_m_v = _parse_metrics(log_text)
                results.append(RunResult(item["name"], item["cmd"], best_acc, cid_m_t, cid_m_v, item["log_path"]))
            running = still_running

            busy = {item["gpu"] for item in running}
            available = _pick_available_gpus(gpus, args.min_free_mb, busy, args.allow_no_query)
            while available and pending:
                if args.sequential and running:
                    break
                gpu = available.pop(0)
                job = pending.pop(0)
                name = job["name"]
                cmd = job["cmd"] + ["--device", str(gpu)]
                log_path = os.path.join(args.logs_dir, f"{name}.log")
                os.makedirs(args.logs_dir, exist_ok=True)
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write(" ".join(cmd) + "\n\n")
                    f.flush()
                    print(f"Running {name} on GPU {gpu} ...")
                    proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
                running.append({"proc": proc, "name": name, "cmd": cmd, "gpu": gpu, "log_path": log_path})

            if pending or running:
                time.sleep(args.poll_interval)

    # Pick the best by main metric (test_acc).
    best = None
    for r in results:
        if r.best_acc is None:
            continue
        if best is None or r.best_acc > best.best_acc:
            best = r

    summary_path = os.path.join(args.logs_dir, "grid_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"{r.name}\tacc={r.best_acc}\tmt={r.cid_m_t_mean}\tmv={r.cid_m_v_mean}\tlog={r.log_path}\n")
        if best is not None:
            f.write("\nBEST\n")
            f.write(f"{best.name}\tacc={best.best_acc}\tmt={best.cid_m_t_mean}\tmv={best.cid_m_v_mean}\tlog={best.log_path}\n")

    if best is None:
        print("No valid runs found. Check logs for errors.")
    else:
        print("Best run:", best.name, "acc=", best.best_acc)
        print("Summary:", summary_path)


if __name__ == "__main__":
    main()
