import argparse
import json
import os
import subprocess
import sys


def _parse_seeds(seeds_arg, seed_file):
    seeds = []
    if seeds_arg:
        for item in seeds_arg.split(","):
            item = item.strip()
            if item:
                seeds.append(int(item))
    if seed_file:
        with open(seed_file, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                seeds.append(int(line))
    # Preserve order, drop duplicates
    return list(dict.fromkeys(seeds))


def _daemonize(log_file):
    if os.name != "posix":
        raise RuntimeError("Daemon mode only supports POSIX systems.")
    if os.environ.get("SEED_RUNNER_DETACHED") == "1":
        return

    pid = os.fork()
    if pid > 0:
        print(f"Seed runner detached (pid={pid}). Logs: {log_file}")
        sys.exit(0)

    os.setsid()
    pid = os.fork()
    if pid > 0:
        os._exit(0)

    os.environ["SEED_RUNNER_DETACHED"] = "1"
    sys.stdout.flush()
    sys.stderr.flush()
    with open(log_file, "a", encoding="utf-8") as log:
        os.dup2(log.fileno(), sys.stdout.fileno())
        os.dup2(log.fileno(), sys.stderr.fileno())


def main():
    parser = argparse.ArgumentParser(description="Run main.py sequentially with multiple seeds.")
    parser.add_argument("--seeds", type=str, default=None,
                        help="comma-separated seeds, e.g. 1,2,3")
    parser.add_argument("--seed_file", type=str, default=None,
                        help="path to file with one seed per line")
    parser.add_argument("--results_dir", type=str, default="../output_dir/seed_runs",
                        help="directory to save per-seed results and summary")
    parser.add_argument("--summary_name", type=str, default="summary_96_9e-4.jsonl",
                        help="summary file name in results_dir")
    parser.add_argument("--detach", action="store_true",
                        help="detach from terminal so closing it won't stop the run")
    parser.add_argument("--log_file", type=str, default=None,
                        help="log file path when using --detach")
    args, passthrough = parser.parse_known_args()

    seeds = _parse_seeds(args.seeds, args.seed_file)
    if not seeds:
        raise ValueError("No seeds provided. Use --seeds or --seed_file.")

    results_dir = os.path.abspath(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)
    summary_path = os.path.join(results_dir, args.summary_name)

    log_file = args.log_file or os.path.join(results_dir, "runner_96_9e-4.log")
    if args.detach:
        _daemonize(log_file)

    src_dir = os.path.dirname(os.path.abspath(__file__))
    main_py = os.path.join(src_dir, "main.py")

    for seed in seeds:
        result_path = os.path.join(results_dir, f"seed_{seed}.json")
        cmd = [
            sys.executable,
            main_py,
            "--seed",
            str(seed),
            "--results_path",
            result_path,
        ] + passthrough
        print(f"Running seed={seed}: {' '.join(cmd)}")
        completed = subprocess.run(cmd, cwd=src_dir)
        record = {
            "seed": seed,
            "returncode": completed.returncode,
            "results_path": result_path,
        }
        if completed.returncode == 0 and os.path.exists(result_path):
            with open(result_path, "r", encoding="utf-8") as fin:
                record["best_test"] = json.load(fin).get("best_test")
        else:
            record["error"] = "run_failed_or_missing_results"
        with open(summary_path, "a", encoding="utf-8") as fout:
            fout.write(json.dumps(record, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
