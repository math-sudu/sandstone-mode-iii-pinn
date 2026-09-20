"""List or execute the input-only package's operator and PINN training steps."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def gauge_stations():
    inputs = ROOT / "results/data/e1_inputs"
    fitted = set(json.loads((inputs / "identification_split.json").read_text())["identification_set"])
    stations = {a: {} for a in (4, 6, 8)}
    with (inputs / "dic_channels/csd_w_cod_curve.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["specimen_id"] in fitted and row["level_kind"] == "measured":
                stations[int(row["a_mm"])].setdefault(row["specimen_id"], float(row["ext_xn_mm"]))
    return stations


def build_operators():
    sys.path.insert(0, str(ROOT / "src"))
    from baxi_pinn.anbd3d_cohesive import build_interface_operators
    directory = ROOT / "results/runs/s1b_field_channel/ops"
    directory.mkdir(parents=True, exist_ok=True)
    for depth, stations in gauge_stations().items():
        target = directory / f"ops_s1b_a{depth}.pkl"
        if target.exists():
            raise FileExistsError(f"Operator already exists: {target}")
        operators = build_interface_operators(float(depth), scale=1.15,
                                             station_map=stations, surface_rows=True)
        with target.open("wb") as handle:
            pickle.dump(operators, handle, protocol=4)
        print(f"Built {target}", flush=True)


def run_stage(stage):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(ROOT / "C-G/src"), str(ROOT / "src")))
    subprocess.run([sys.executable, str(ROOT / stage["script"]),
                    "--config", str(ROOT / stage["config"])],
                   cwd=ROOT / "C-G", env=environment, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--operators", action="store_true")
    actions.add_argument("--run", metavar="STAGE")
    actions.add_argument("--all", action="store_true", help="Build operators and train all stages")
    args = parser.parse_args()
    plan = json.loads((ROOT / "reproduction.json").read_text())
    if args.operators or args.all:
        build_operators()
    if args.run or args.all:
        stages = [s for s in plan["stages"] if args.all or s["name"] == args.run]
        if not stages:
            parser.error(f"Unknown training stage: {args.run}")
        for stage in stages:
            run_stage(stage)
    elif not args.operators:
        print("python reproduce.py --operators")
        for stage in plan["stages"]:
            print(f"python reproduce.py --run {stage['name']}")


if __name__ == "__main__":
    main()
