import argparse
import hashlib
import importlib.metadata
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean

from scipy.stats import binomtest
from wandb.proto.wandb_internal_pb2 import Record
from wandb.sdk.internal.datastore import DataStore


SUCCESS_KEYS = ("task_success", "physical_success")
PENETRATION_KEYS = ("all", "block_dustpan", "robot_self", "forbidden_collision")
RUN_KEYS = ("id", "method", "condition", "seed", "role")
GROUP_KEYS = ("method", "condition", "role", "domain")


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path, sources: dict[str, str]) -> dict:
    sources[str(path.resolve())] = file_hash(path)
    return json.loads(path.read_text())


def read_history(path: Path) -> list[dict]:
    store = DataStore()
    store.open_for_scan(str(path))
    points = []
    try:
        while (payload := store.scan_data()) is not None:
            record = Record()
            record.ParseFromString(payload)
            if not record.HasField("history"):
                continue
            fields = {item.key or "/".join(item.nested_key): json.loads(item.value_json)
                      for item in record.history.item}
            if "train/steps" in fields:
                points.append({"step": fields["train/steps"], "loss": fields["train/loss"],
                               "update_s": fields["train/update_s"],
                               "data_s": fields["train/dataloading_s"],
                               "timestamp_unix": fields["_timestamp"]})
    finally:
        store.close()
    return points


def read_training(config: dict, base: Path, sources: dict[str, str]) -> dict:
    audit = read_json(base / config["attempt"], sources)
    path = base / config["wandb"]
    sources[str(path.resolve())] = file_hash(path)
    return {"audit": audit, "points": read_history(path),
            "nominal_log_interval_updates": config["log_interval"],
            "point_semantics": "loss, update_s and data_s are means since the previous log reset"}


def read_evaluation(directory: Path, sources: dict[str, str]) -> tuple[list[dict], dict]:
    audit = read_json(directory / "evaluation.json", sources)
    path = directory / "results.jsonl"
    sources[str(path.resolve())] = file_hash(path)
    results = [json.loads(line) for line in path.read_text().splitlines()]
    metrics = [read_json(directory / f"{index:05d}" / "metrics.json", sources)
               for index in range(len(results))]
    return metrics, audit


def success_rate(values: list[bool]) -> dict:
    successes, total = sum(values), len(values)
    interval = binomtest(successes, total).proportion_ci(confidence_level=0.95, method="wilson")
    return {"successes": successes, "total": total, "rate": successes / total,
            "wilson95": [interval.low, interval.high]}


def value_range(values: list[float]) -> dict:
    return {"mean": fmean(values), "min": min(values), "max": max(values)}


def summarize_cases(metrics: list[dict]) -> dict:
    blocks = [row["blocks_in_dustpan"] for row in metrics]
    penetration = {key: [row["forbidden_collision_penetration_m"] if key == "forbidden_collision"
                         else row["max_penetration_m"][key] for row in metrics]
                   for key in PENETRATION_KEYS}
    return {
        "scene_count": len(metrics),
        "scene_sha256": [row["scene_sha256"] for row in metrics],
        **{key: success_rate([row[key] for row in metrics]) for key in SUCCESS_KEYS},
        "collected_blocks": {**value_range(blocks), "total": sum(blocks),
                             "available": sum(row["block_count"] for row in metrics)},
        "penetration_m": {key: {"mean": fmean(values), "max": max(values)}
                          for key, values in penetration.items()},
    }


def across_seeds(per_seed: list[dict], expected_seeds: list[int]) -> list[dict]:
    groups = defaultdict(list)
    for row in per_seed:
        groups[tuple(row[key] for key in GROUP_KEYS)].append(row)
    summaries = []
    for key, rows in groups.items():
        summaries.append({
            **dict(zip(GROUP_KEYS, key)),
            "run_ids": [row["run_id"] for row in rows],
            "seeds": sorted(row["seed"] for row in rows), "seed_count": len(rows),
            "expected_seeds": expected_seeds,
            **{f"{name}_rate": value_range([row[name]["rate"] for row in rows])
               for name in SUCCESS_KEYS},
            "collected_blocks_mean": value_range([row["collected_blocks"]["mean"] for row in rows]),
            "penetration_m": {
                name: {f"per_seed_{stat}": value_range([row["penetration_m"][name][stat]
                                                       for row in rows])
                       for stat in ("mean", "max")} for name in PENETRATION_KEYS},
            "aggregation": "equal-weight seed mean and observed range; no pooled confidence interval",
        })
    return summaries


def paired_outcomes(reference: list[dict], treatment: list[dict]) -> dict:
    left = {row["scene_sha256"]: row for row in reference}
    right = {row["scene_sha256"]: row for row in treatment}
    common = sorted(left.keys() & right.keys())
    outcomes = {}
    for key in SUCCESS_KEYS:
        table = [[0, 0], [0, 0]]
        for scene in common:
            table[int(left[scene][key])][int(right[scene][key])] += 1
        outcomes[key] = {"table": table,
                         "reference_rate": sum(table[1]) / len(common),
                         "treatment_rate": (table[0][1] + table[1][1]) / len(common),
                         "rate_difference": (table[0][1] - table[1][0]) / len(common)}
    return {"common_count": len(common), "common_scene_sha256": common,
            "reference_only": sorted(left.keys() - right.keys()),
            "treatment_only": sorted(right.keys() - left.keys()),
            "table_axes": {"rows": "reference", "columns": "treatment", "order": [False, True]},
            **outcomes}


def summarize(registry_path: Path) -> dict:
    sources = {}
    registry = read_json(registry_path, sources)
    base = registry_path.parent
    training, per_seed, metrics_by_run = [], [], {}
    for run in registry["runs"]:
        identity = {key: run[key] for key in RUN_KEYS}
        if run["training"] is not None:
            training.append({**identity, **read_training(run["training"], base, sources)})
        for evaluation in run["evaluations"]:
            metrics, audits = [], []
            for directory in evaluation["directories"]:
                cases, audit = read_evaluation(base / directory, sources)
                metrics.extend(cases)
                audits.append(audit)
            metrics_by_run[run["id"], evaluation["domain"]] = metrics
            per_seed.append({"run_id": run["id"], **{k: run[k] for k in RUN_KEYS[1:]},
                             "domain": evaluation["domain"], "audits": audits,
                             **summarize_cases(metrics)})
    pairs = [{**pair, **paired_outcomes(metrics_by_run[pair["reference"], pair["domain"]],
                                        metrics_by_run[pair["treatment"], pair["domain"]])}
             for pair in registry["pairs"]]
    return {"schema_version": 2, "training": training, "per_seed": per_seed,
            "across_seeds": across_seeds(per_seed, registry["expected_seeds"]), "paired": pairs,
            "sources_sha256": sources, "implementation_sha256": file_hash(Path(__file__)),
            "packages": {name: importlib.metadata.version(name) for name in ("wandb", "scipy")}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.registry)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
