"""
Run a DP sweep with N processes in parallel on the same GPU.

Splits the (mechanism, epsilon, delta) grid across N worker processes,
each limited to memory_gb / N of GPU memory.

Two targets:
  --target utility : runs scripts/train_with_dp.py (seq2seq classifier)
  --target reid    : runs scripts/train_reid.py (re-identification CNN)

Usage:
    # 3 parallel utility workers, each ~7 GB of the 24 GB on the 3090 Ti
    python scripts/run_parallel_sweep.py --target utility --workers 3 --epochs 200

    # 3 parallel Re-ID workers (adaptive attacker)
    python scripts/run_parallel_sweep.py --target reid --workers 3 --epochs 50

    # 2 workers, more conservative memory
    python scripts/run_parallel_sweep.py --target utility --workers 2 --gpu-memory-gb 10

Each worker writes per-config metrics under results/{dp,reid/dp}/...,
so --skip-existing works if a sweep is interrupted and restarted.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs import DP


def shard_list(items, n_shards: int, shard_id: int):
    """Return every n_shards-th element starting at shard_id."""
    return [x for i, x in enumerate(items) if i % n_shards == shard_id]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--target", choices=["utility", "reid"], default="utility",
                   help="Which sweep to run: utility classifier or Re-ID attacker.")
    p.add_argument("--workers", type=int, default=3,
                   help="Number of parallel processes (default 3)")
    p.add_argument("--gpu-memory-gb", type=float, default=None,
                   help="GPU memory per worker (default: 22.0 / workers)")
    p.add_argument("--mechanisms", nargs="+",
                   default=list(DP.mechanisms))
    p.add_argument("--epsilons", type=float, nargs="+", default=None)
    p.add_argument("--deltas", type=float, nargs="+", default=None,
                   help="If omitted, downstream scripts use DP.delta_for(mech).")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="One run per seed (default: configs.DP.seeds)")
    p.add_argument("--epochs", type=int, default=None,
                   help="Default: 200 for utility, 50 for reid")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-every", type=int, default=20)
    # Utility-specific
    p.add_argument("--data-source", choices=["wfdb", "mat"], default="wfdb")
    p.add_argument("--mat-path", type=str, default=None)
    # Re-ID specific
    p.add_argument("--reid-dataset", choices=["mitbih", "ecgid"], default="mitbih",
                   help="Re-ID database to use.")
    p.add_argument("--reid-data-split", choices=["ds1", "all"], default="ds1",
                   help="MIT-BIH only: ds1 = 22 patients, all = 44 patients.")
    p.add_argument("--reid-threat-model",
                   choices=["clean_attacker", "adaptive_attacker", "both"],
                   default="adaptive_attacker")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Default: results/dp for utility, results/reid for reid")
    p.add_argument("--skip-existing", action="store_true", default=True)
    args = p.parse_args()

    # Resolve defaults that depend on --target and --reid-dataset
    if args.epochs is None:
        args.epochs = 200 if args.target == "utility" else 50
    if args.out_dir is None:
        if args.target == "utility":
            args.out_dir = "results/dp"
        elif args.reid_dataset == "ecgid":
            args.out_dir = "results/reid_ecgid"
        else:
            args.out_dir = "results/reid"

    n = args.workers
    gpu_mem = args.gpu_memory_gb if args.gpu_memory_gb is not None else (22.0 / n)

    epsilons = args.epsilons if args.epsilons else list(DP.epsilons)
    seeds = args.seeds if args.seeds else list(DP.seeds)
    deltas = args.deltas  # None means: let downstream pick per mech

    print(f"Parallel DP sweep")
    print(f"  Target:            {args.target}")
    print(f"  Workers:           {n}")
    print(f"  GPU memory/worker: {gpu_mem:.1f} GiB")
    print(f"  Mechanisms:        {args.mechanisms}")
    print(f"  Epsilons:          {epsilons}")
    if deltas is None:
        print(f"  Deltas:            per-mechanism (DP.delta_for)")
    else:
        print(f"  Deltas:            {deltas}")
    print(f"  Seeds:             {seeds}")
    print(f"  Epochs:            {args.epochs}")
    if args.target == "reid":
        print(f"  Re-ID data split:  {args.reid_data_split}")
        print(f"  Threat model:      {args.reid_threat_model}")
    n_eff_deltas = len(deltas) if deltas is not None else 1
    n_total = len(args.mechanisms) * len(epsilons) * n_eff_deltas * len(seeds)
    if args.target == "reid" and args.reid_threat_model == "both":
        n_total *= 2
    print(f"  Total configs:     ~{n_total} (before invalid-skip)")
    print(f"  Per worker:        ~{n_total // n} configs")
    print()

    log_dir = Path(args.out_dir) / "parallel_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    procs = []
    for shard_id in range(n):
        eps_slice = shard_list(epsilons, n, shard_id)
        if not eps_slice:
            continue

        if args.target == "utility":
            cmd = [
                sys.executable, "scripts/train_with_dp.py",
                "--gpu-memory-gb", str(gpu_mem),
                "--mechanisms", *args.mechanisms,
                "--epsilons", *[str(e) for e in eps_slice],
                "--seeds", *[str(s) for s in seeds],
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--eval-every", str(args.eval_every),
                "--data-source", args.data_source,
                "--out-dir", args.out_dir,
            ]
            if deltas is not None:
                cmd.extend(["--deltas", *[str(d) for d in deltas]])
            if args.mat_path:
                cmd.extend(["--mat-path", args.mat_path])
        else:  # reid
            cmd = [
                sys.executable, "scripts/train_reid.py",
                "--gpu-memory-gb", str(gpu_mem),
                "--dp-sweep",
                "--dataset", args.reid_dataset,
                "--data-split", args.reid_data_split,
                "--threat-model", args.reid_threat_model,
                "--mechanisms", *args.mechanisms,
                "--epsilons", *[str(e) for e in eps_slice],
                "--seeds", *[str(s) for s in seeds],
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--out-dir", args.out_dir,
            ]
            if deltas is not None:
                cmd.extend(["--deltas", *[str(d) for d in deltas]])
        if args.skip_existing:
            cmd.append("--skip-existing")

        log_path = log_dir / f"worker_{shard_id}.log"
        print(f"[worker {shard_id}] eps={eps_slice}, log={log_path}")
        log_fh = open(log_path, "w")
        proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
        procs.append((shard_id, proc, log_fh))

    print()
    print(f"Started {len(procs)} workers. Monitoring (poll every 30s)...")
    print(f"Tail logs with: tail -f {log_dir}/worker_*.log")
    print()

    while procs:
        time.sleep(30)
        still = []
        for shard_id, proc, log_fh in procs:
            rc = proc.poll()
            if rc is None:
                still.append((shard_id, proc, log_fh))
            else:
                log_fh.close()
                status = "OK" if rc == 0 else f"FAILED (exit {rc})"
                print(f"[worker {shard_id}] finished: {status}")
        procs = still
        if procs:
            print(f"  Still running: {[s for s, _, _ in procs]}")

    print()
    print("All workers done")


if __name__ == "__main__":
    main()
