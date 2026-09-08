import os

for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[variable] = "1"

import argparse
import concurrent.futures
import json
from pathlib import Path

from src.augmentation import generate_episode, sample_variants
from src.simulation import resolve_scene_spec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=Path("agent/out/scene_spec.json"))
    parser.add_argument("--assets", type=Path, default=Path("thirdparty/mujoco_menagerie/pal_tiago_dual"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=16, help="Power-of-two Sobol candidate count")
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--scale", type=float, default=0.25)
    parser.add_argument("--split", default="train")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--target-successes", type=int)
    args = parser.parse_args()
    variants = sample_variants(resolve_scene_spec(args.spec), args.count, args.seed, args.scale, args.split)
    args.output.mkdir(parents=True, exist_ok=False)
    print(json.dumps({"pid": os.getpid(), "output": str(args.output),
                      "candidates": args.count, "workers": args.workers}), flush=True)
    folders = [args.output / f"episode-{index:06d}" for index in range(len(variants))]
    with (args.output / "proposed.jsonl").open("w") as stream:
        for variant, folder in zip(variants, folders):
            stream.write(json.dumps({"status": "proposed", "episode_dir": folder.name,
                                     "variant": variant.to_dict()}) + "\n")
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = pool.map(generate_episode, variants, [args.assets] * len(variants), folders)
        accepted = []
        with (args.output / "attempts.jsonl").open("w") as stream:
            for folder, result in zip(folders, results):
                result["episode_dir"] = folder.name
                stream.write(json.dumps(result) + "\n")
                stream.flush()
                if result["success"]:
                    accepted.append(folder.name)
                print(json.dumps({"episode": folder.name, "success": result["success"],
                                  "collected": result["metrics"]["blocks_in_dustpan"],
                                  "checks": result["validity_checks"]}), flush=True)
    (args.output / "successful.json").write_text(json.dumps(accepted, indent=2))
    if args.target_successes is not None:
        selected = accepted[:args.target_successes]
        (args.output / "training-episodes.json").write_text(json.dumps(selected, indent=2))
        (args.output / "training-selection.json").write_text(json.dumps({
            "requested": args.target_successes, "selected": len(selected),
            "all_successes": len(accepted), "rule": "first passing candidates in proposal order",
        }, indent=2))


if __name__ == "__main__":
    main()
