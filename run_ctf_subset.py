"""Run a hand-picked subset of CTF tasks on chosen models, pass@k style.

The authors' batch runner always runs every task for every model in
config/model.yaml.  With free-tier API limits that is far too many calls, so
this runner takes an explicit task list.

Usage:
    python run_ctf_subset.py --tasks subset.txt --models my-model-a my-model-b \
        --attempts 3 --jobs 2

subset.txt: one task path per line, e.g. data/CTF/04-RSA/01-blue-hens-2023
Attempt i is stored under outputs/CTF-<run-prefix><i>/ so attempts never collide.
Results are appended to outputs/subset_results.jsonl and are resumable: a
(model, task, attempt) triple that is already recorded is skipped.
"""

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
from pathlib import Path

RESULTS = Path("outputs/subset_results.jsonl")


def load_done() -> set:
    done = set()
    if RESULTS.exists():
        for line in RESULTS.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["model"], r["task"], r["attempt"]))
    return done


def run_one(model: str, task: str, attempt: int, prefix: str, max_iter: int) -> dict:
    env = os.environ.copy()
    for v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        env.setdefault(v, "1")
    cmd = [
        sys.executable, "run_single_ctf_task.py",
        "--model", model, "--task-path", task,
        "--max-iterations", str(max_iter),
        "--id", f"{prefix}{attempt}",
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    result: object = False
    err = ""
    lines = p.stdout.strip().splitlines()
    if lines:
        try:
            summary = json.loads(lines[-1])
            result = bool(summary.get("result"))
            err = summary.get("error", "")
        except json.JSONDecodeError:
            err = "unparseable output"
    elif p.returncode != 0:
        err = p.stderr.strip()[:300]
    return {"model": model, "task": task, "attempt": attempt,
            "solved": result is True, "error": err}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--max-iterations", type=int, default=100)
    ap.add_argument("--run-prefix", default="sub")
    a = ap.parse_args()

    tasks = [t.strip() for t in Path(a.tasks).read_text().splitlines()
             if t.strip() and not t.startswith("#")]
    done = load_done()
    todo = [(m, t, i) for m in a.models for t in tasks for i in range(a.attempts)
            if (m, t, i) not in done]
    print(f"{len(todo)} runs to do ({len(done)} already recorded)")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with cf.ThreadPoolExecutor(max_workers=a.jobs) as ex, RESULTS.open("a") as out:
        futs = {ex.submit(run_one, m, t, i, a.run_prefix, a.max_iterations): (m, t, i)
                for m, t, i in todo}
        for f in cf.as_completed(futs):
            r = f.result()
            out.write(json.dumps(r) + "\n")
            out.flush()
            print(r)

    # pass@k summary
    rows = [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]
    for m in a.models:
        solved = {r["task"] for r in rows if r["model"] == m and r["solved"]}
        print(f"{m}: pass@{a.attempts} = {len(solved)}/{len(tasks)}")


if __name__ == "__main__":
    main()
