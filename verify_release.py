"""Check archive integrity, measured-input coverage and recomputation dependencies."""
from __future__ import annotations

import argparse
import csv
import io
import json
import posixpath
from pathlib import Path
import re
import zipfile

PREFIX = "CG_Data_Code/"


def check(read, names):
    for name in names:
        if name.endswith(".py"):
            compile(read(name), name, "exec")
    split = json.loads(read("results/data/e1_inputs/identification_split.json"))
    fitted, assessed = set(split["identification_set"]), set(split["held_out_set"])
    assert len(fitted) == 18 and len(assessed) == 9 and fitted.isdisjoint(assessed)
    peaks = list(csv.DictReader(io.StringIO(read(
        "results/data/group_assets/f723fea9/curated/force_all_specimens.csv").decode())))
    assert fitted | assessed <= {r["specimen"] for r in peaks}
    gauges = list(csv.DictReader(io.StringIO(read(
        "results/data/e1_inputs/dic_channels/csd_w_cod_curve.csv").decode())))
    assert {int(row["beta_deg"]) for row in gauges} == {0, 45, 90}
    plan = json.loads(read("reproduction.json"))
    preceding = set()
    generated = {f"results/runs/s1b_field_channel/ops/ops_s1b_a{a}.pkl" for a in (4, 6, 8)}

    def check_paths(value, key=""):
        if key == "output_directory":
            return
        if isinstance(value, dict):
            for k, v in value.items():
                check_paths(v, k)
        elif isinstance(value, list):
            for v in value:
                check_paths(v, key)
        elif isinstance(value, str) and ("/" in value or key.endswith("_path")):
            target = posixpath.normpath("C-G/" + value)
            match = re.search(r"experiments/interface_pinn/runs/([^/]+)", value)
            if match:
                assert match[1] in preceding, f"Initialization is not generated first: {value}"
            else:
                assert target in names or target in generated, f"Missing input: {target}"

    for stage in plan["stages"]:
        assert stage["script"] in names and stage["config"] in names
        assert set(stage["dependencies"]) <= preceding
        check_paths(json.loads(read(stage["config"])))
        preceding.add(stage["name"])
    assert set(plan["selected_runs"].values()) <= preceding
    assert not any("/runs/" in name or name.endswith((".pt", ".pkl", ".npz", ".pdf", ".svg", ".jsonl"))
                   for name in names), "Generated result included in the input-only package"
    return {"files": len(names), "notched_specimens": 27,
            "initialization_and_fit_stages": len(preceding),
            "archive_contents": "Source, experimental inputs and parameters only"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    if args.archive:
        with zipfile.ZipFile(args.archive) as archive:
            assert archive.testzip() is None, "Archive CRC error"
            paths = archive.namelist()
            assert len(paths) == len(set(paths)) and all(p.startswith(PREFIX) for p in paths)
            result = check(lambda name: archive.read(PREFIX + name),
                           {p[len(PREFIX):] for p in paths})
    else:
        root = Path(__file__).resolve().parent
        names = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
                 and "__pycache__" not in p.parts and ".git" not in p.parts and "runs" not in p.parts}
        result = check(lambda name: (root / name).read_bytes(), names)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
