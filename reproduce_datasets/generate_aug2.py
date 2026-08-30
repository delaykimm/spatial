"""Regenerates aug2_images/ + aug2_render_manifest_disjoint.json.

Unlike the other 3 scripts in this folder (pure matplotlib, no external dependency),
aug2 is rendered with real Blender-textured 3D objects via the external FORG3D toolkit --
this script drives that toolkit and copies its output in, it does NOT reimplement
rendering itself. REQUIRES, on the machine this was built on (paths below are that
server's actual install, not portable elsewhere without adjusting them):
  - Blender 4.3.2 at /node_data/urp26su_jiyun/tools/blender-4.3.2-linux-x64/blender
  - The FORG3D toolkit at /node_data/urp26su_jiyun/tools/FORG3D/ (objaverse object
    library + render_scene.py). Its own src/config.json hardcodes a FIXED output
    location (ours/results/forg3d_images/{images,scenes,masks}/), shared across
    whatever else uses that toolkit -- this script renders there (can't redirect
    without editing that shared config) and then copies just the pairs it rendered
    into --out.
  - >=1 CUDA GPU (Cycles). Renders each (obj1, obj2, direction) triple as its own
    `blender --background` subprocess, round-robined across GPUS below.

807 images total (per the current manifest) -- each is a real Blender render, so this is
slow (subprocess + GPU render per image, not a bulk operation) and will contend for GPU
with anything else running on this shared box.

Object-pair partition: every object pair is assigned to exactly one of
horizontal/vertical/closefar (never reused across axes) --
closefar takes every eligible non-same-group pair that isn't small+large, vertical takes
all same-group pairs (the only kind it can use), horizontal gets the small+large leftovers.

Usage: python generate_aug2.py [--out DIR] [--out-json PATH]
"""
import argparse
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

FORG3D_PROPERTIES_PATH = "/node_data/urp26su_jiyun/tools/FORG3D/data/objaverse/properties.json"
BLENDER = "/node_data/urp26su_jiyun/tools/blender-4.3.2-linux-x64/blender"
RENDER_SCRIPT = "/node_data/urp26su_jiyun/tools/FORG3D/src/render_scene.py"
SCRIPTS_DIR = "/node_data/urp26su_jiyun/tools/FORG3D/scripts"
FORG3D_OUT = Path("/node_data/urp26su_jiyun/ours/results/forg3d_images")  # fixed by FORG3D's own config.json

GPUS = [2, 5, 6]  # adjust to whichever GPUs have free memory -- check nvidia-smi first
DIRECTIONS = {"horizontal": ("right", "left"), "vertical": ("above", "below"), "closefar": ("front", "behind")}
CAMERA_ARGS = ["--camera-tilt", "85", "--camera-pan", "45",
               "--camera-height", "2", "--camera-focal-length", "60"]


def disjoint_pairs_for_axes():
    """closefar first (every non-same-group, non-small+large pair), then vertical (all
    same-group pairs), horizontal gets the small+large leftovers. No object pair repeats
    across axes."""
    with open(FORG3D_PROPERTIES_PATH) as f:
        props = json.load(f)
    objs = sorted(props.keys())
    all_pairs = list(combinations(objs, 2))
    group = {n: props[n]["group"] for n in objs}

    closefar = [p for p in all_pairs
                if group[p[0]] != group[p[1]] and {group[p[0]], group[p[1]]} != {"small", "large"}]
    remaining = [p for p in all_pairs if p not in set(closefar)]
    vertical = [p for p in remaining if group[p[0]] == group[p[1]]]
    horizontal = [p for p in remaining if p not in set(vertical)]
    return {"horizontal": horizontal, "vertical": vertical, "closefar": closefar}


def render_one(obj1, obj2, direction, gpu_id):
    """Renders (obj1=ground, obj2=figure, direction) on the given GPU. Output lands in
    FORG3D_OUT/images (plus matching masks/scenes) at {obj2}_{direction}_{obj1}.png --
    skipped if already there. Returns that filename stem, or None if FORG3D itself
    skipped it (overlap detected) or errored."""
    stem = f"{obj2}_{direction}_{obj1}"
    out_path = FORG3D_OUT / "images" / f"{stem}.png"
    if out_path.exists():
        return stem
    cmd = [BLENDER, "--background", "--python", RENDER_SCRIPT, "--",
           "--objects", obj1, obj2, "--direction", direction,
           "--filename-prefix", "unused", *CAMERA_ARGS]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    result = subprocess.run(cmd, cwd=SCRIPTS_DIR, capture_output=True, text=True, timeout=60, env=env)
    if out_path.exists():
        return stem
    if "Overlap detected" in result.stdout:
        return None
    print(f"  WARNING: no output for {obj1}/{direction}/{obj2}\n{result.stdout[-500:]}\n{result.stderr[-500:]}")
    return None


def _copy_render(stem, sub, ext, out_dir):
    src = FORG3D_OUT / sub / f"{stem}{ext}"
    dst = out_dir / sub / f"{stem}{ext}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copy2(src, dst)


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--out", default="../data/aug2_images")
    ap_.add_argument("--out-json", default="../data/aug2_render_manifest_disjoint.json")
    args = ap_.parse_args()
    out_dir = Path(args.out)

    pairs_by_axis = disjoint_pairs_for_axes()
    manifest = {}

    for axis in ["horizontal", "vertical", "closefar"]:
        pairs = pairs_by_axis[axis]
        plus_dir, minus_dir = DIRECTIONS[axis]
        print(f"\n[{axis}] rendering {len(pairs)} disjoint pairs x 2 directions ({plus_dir}/{minus_dir}) "
              f"on {len(GPUS)} GPUs {GPUS}")

        jobs = [(o1, o2, d) for o1, o2 in pairs for d in (plus_dir, minus_dir)]
        results = {}
        with ThreadPoolExecutor(max_workers=len(GPUS)) as pool:
            futures = {pool.submit(render_one, o1, o2, d, GPUS[i % len(GPUS)]): (o1, o2, d)
                       for i, (o1, o2, d) in enumerate(jobs)}
            done = 0
            for fut in as_completed(futures):
                o1, o2, d = futures[fut]
                results[(o1, o2, d)] = fut.result()
                done += 1
                if done % 30 == 0:
                    print(f"  {done}/{len(jobs)} renders done")

        axis_manifest, rendered, skipped = [], 0, 0
        for obj1, obj2 in pairs:
            stem_plus = results.get((obj1, obj2, plus_dir))
            stem_minus = results.get((obj1, obj2, minus_dir))
            if stem_plus and stem_minus:
                for stem in (stem_plus, stem_minus):
                    for sub, ext in [("images", ".png"), ("masks", ".png"), ("scenes", ".json")]:
                        _copy_render(stem, sub, ext, out_dir)
                axis_manifest.append({"obj1": obj1, "obj2": obj2,
                                       "plus_image": f"aug2_images/images/{stem_plus}.png",
                                       "minus_image": f"aug2_images/images/{stem_minus}.png"})
                rendered += 1
            else:
                skipped += 1
        manifest[axis] = axis_manifest
        print(f"[{axis}] done: {rendered} usable pairs, {skipped} skipped (overlap or error)")

    with open(args.out_json, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nsaved manifest -> {args.out_json}")


if __name__ == "__main__":
    main()
