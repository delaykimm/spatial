"""Genuine multi-hop MLLM generation test, MINED FROM REAL CLEVR PHOTOS -- same idea as
multihop_referring_pipeline.py (resolve a nested referring expression to find Z, then tell
it apart from Y, one hop short) but the objects come from data/clevr_val_scenes.json
instead of being rendered.

Unlike clevr_chain_pipeline.py (which picks N random objects in random order -- fine for
the additive-composition/latent test, which doesn't care about referring-phrase
ambiguity), THIS pipeline needs "next object in that direction" to be unambiguous, so it
picks N random objects from a scene and SORTS them by their real axis coordinate (rank
0..N-1) before doing the rank walk -- same construction as the synthetic version, just
using real (already-scattered) CLEVR positions instead of a controlled sorted layout.

Usage:
    python clevr_multihop_referring_pipeline.py --model qwen3vl
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import json
import random

import numpy as np
from PIL import Image

import triplet_pipeline as cxp
import axis_pipeline as ap
import chain_hop_pipeline as chp
import clevr_chain_pipeline as ccp
import multihop_referring_pipeline as mrp

_CFG = ap.CFG["multihop_referring"]
AXES = ["horizontal", "closefar"]
AXIS_DIRS = ccp.AXIS_DIRS
HOPS = _CFG["hops"]
N_SCENES_PER_HOP = _CFG["n_scenes_per_hop"]
SEED = {"horizontal": 12042, "closefar": 12242}
DIR_PHRASE = _CFG["dir_phrase"]

CHAIN_IMAGES_DIR = f"{ap.DATA}/clevr_multihop_referring_images"
CHAIN_MANIFEST_PATH = f"{ap.DATA}/clevr_multihop_referring_manifest.json"
CHAIN_SHARD_BASE = f"{ap.RESULTS}/shards/clevr_multihop_referring_shard.npz"


def chain_shard_out_path():
    return ap.suffixed(CHAIN_SHARD_BASE)


def chain_hop_build_manifest():
    with open(cxp.CLEVR_SCENES_PATH) as f:
        scenes = json.load(f)["scenes"]

    manifest = {}
    for axis in AXES:
        axis_dir_name = AXIS_DIRS[axis]
        rng = random.Random(SEED[axis])
        scene_order = list(range(len(scenes)))
        rng.shuffle(scene_order)

        per_hop = {}
        si_ptr = 0
        for hops in HOPS:
            n_objects = hops + 1
            scenes_out = []
            while len(scenes_out) < N_SCENES_PER_HOP and si_ptr < len(scene_order):
                s = scenes[scene_order[si_ptr]]
                si_ptr += 1
                objs = s["objects"]
                if len(objs) < n_objects:
                    continue
                axis_dir = s["directions"][axis_dir_name]
                chosen = rng.sample(range(len(objs)), n_objects)
                descs = [cxp.clevr_desc(objs[i]) for i in chosen]
                if len(set(descs)) < n_objects:
                    continue
                scalars = [cxp.dot(objs[i]["3d_coords"], axis_dir) for i in chosen]
                # sort chosen objects by their real axis coordinate -> rank 0..n-1, so
                # "next object in that direction" is unambiguous, same as the synthetic version
                order = sorted(range(n_objects), key=lambda i: scalars[i])
                descs_by_rank = [descs[i] for i in order]
                scalars_by_rank = [scalars[i] for i in order]

                walk_ranks = mrp.rank_walk(n_objects, hops, rng)
                # reject Z==A (no defined answer) -- see multihop_referring_pipeline.build_scene
                if walk_ranks is None or walk_ranks[-1] == walk_ranks[0]:
                    continue
                phrase = mrp.referring_phrase(axis, walk_ranks,
                                               [{"desc": d} for d in descs_by_rank])
                z_rank, y_rank = walk_ranks[-1], walk_ranks[-2]
                scenes_out.append({
                    "image_path": s["image_filename"], "axis": axis, "hops": hops,
                    "walk_ranks": walk_ranks, "phrase": phrase,
                    "z_desc": descs_by_rank[z_rank], "y_desc": descs_by_rank[y_rank],
                    "objects_by_rank": [{"desc": d, "scalar": sc}
                                        for d, sc in zip(descs_by_rank, scalars_by_rank)],
                })
            per_hop[str(hops)] = scenes_out
            print(f"[clevr_multihop_referring|{axis}|hop{hops}] {len(scenes_out)}/{N_SCENES_PER_HOP} chains found "
                  f"(scanned {si_ptr}/{len(scene_order)} scenes so far)")
        manifest[axis] = per_hop
    with open(CHAIN_MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"saved -> {CHAIN_MANIFEST_PATH}")
    return manifest


def chain_hop_load_or_build_manifest():
    return cxp.load_or_build_json(CHAIN_MANIFEST_PATH, chain_hop_build_manifest, label="clevr_multihop_referring")


def copy_images(manifest, clevr_val_dir):
    from pathlib import Path
    import shutil
    Path(CHAIN_IMAGES_DIR).mkdir(parents=True, exist_ok=True)
    needed = sorted({s["image_path"] for axis in manifest for hops in manifest[axis] for s in manifest[axis][hops]})
    copied, already, missing = 0, 0, []
    for fname in needed:
        dst = Path(CHAIN_IMAGES_DIR) / fname
        if dst.exists():
            already += 1
            continue
        src = Path(clevr_val_dir) / fname
        if not src.exists():
            missing.append(fname)
            continue
        shutil.copy2(src, dst)
        copied += 1
    print(f"copied {copied} new images (+{already} already present) -> {CHAIN_IMAGES_DIR}")
    if missing:
        raise FileNotFoundError(f"{len(missing)} referenced images not found under {clevr_val_dir}, "
                                 f"e.g. {missing[:5]}")


def build_shard(model_key):
    ap.set_model(model_key)
    manifest = chain_hop_load_or_build_manifest()
    model, proc = ap.load_model()

    keys, flat = [], {}
    n_total = sum(len(manifest[axis][str(h)]) for axis in AXES for h in HOPS)
    n_done = 0
    for axis in AXES:
        for hops in HOPS:
            scenes = manifest[axis][str(hops)]
            for idx, s in enumerate(scenes):
                img = Image.open(os.path.join(CHAIN_IMAGES_DIR, s["image_path"])).convert("RGB")
                descs = [s["objects_by_rank"][r]["desc"] for r in s["walk_ranks"]]
                legs = chp.extract_chain(model, proc, img, descs)
                key = f"{axis}|{hops}|{idx}"
                keys.append(key)
                for i, leg in enumerate(legs):
                    flat[f"{key}|leg{i}"] = leg
                n_done += 1
                if n_done % 20 == 0:
                    print(f"  {n_done}/{n_total} chains extracted")

    out_path = chain_shard_out_path()
    np.savez(out_path, keys=np.array(keys, dtype=object), **flat)
    print(f"saved -> {out_path}")
    return out_path


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--model", choices=ap.MODELS, default=ap.DEFAULT_MODEL)
    cli.add_argument("--clevr-val-dir", default=None)
    cli.add_argument("--render-only", action="store_true")
    args = cli.parse_args()

    manifest = chain_hop_load_or_build_manifest()
    if args.clevr_val_dir:
        copy_images(manifest, args.clevr_val_dir)
    elif not os.path.exists(CHAIN_IMAGES_DIR):
        raise SystemExit("no --clevr-val-dir given and no cached images yet -- pass --clevr-val-dir once")

    if not args.render_only:
        build_shard(args.model)
