"""Gets data/clevr_cross_axis_images/ from a local copy of the official CLEVR dataset.

clevr_cross_axis_images is NOT reproducible by code -- it's real rendered photos from the
CLEVR v1.0 dataset (Johnson et al., cs.stanford.edu/people/jcjohns/clevr), not something
this repo generates. What IS already in this repo (data/clevr_val_scenes.json,
data/clevr_cross_axis_triplets.json) is metadata -- ground-truth 3D coordinates for the
official CLEVR val scenes, and this pipeline's own selection of which 496 of those 15,000
val images it actually uses (triplet_pipeline.clevr_sample_triplets(), seeded, already
run once to produce clevr_cross_axis_triplets.json).

So reproducing the images is just: get the official CLEVR val image set locally, then
copy out only the 496 files this pipeline references (not all 15,000) -- what this
script does.

Steps:
  1. Download CLEVR v1.0 from https://cs.stanford.edu/people/jcjohns/clevr/ (the val
     split's images -- filenames look like CLEVR_val_000072.png) and extract it
     somewhere.
  2. python fetch_clevr_images.py --clevr-val-dir /path/to/CLEVR_v1.0/images/val

Only copies files, doesn't touch clevr_val_scenes.json/clevr_cross_axis_triplets.json --
those are already committed and don't need regenerating (same seed -> same 496 files
would be picked again by clevr_sample_triplets() if you ever needed to redo the
selection itself).
"""
import argparse
import json
import shutil
from pathlib import Path

TRIPLETS_PATH = "../data/clevr_cross_axis_triplets.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clevr-val-dir", required=True, help="path to CLEVR_v1.0/images/val/")
    ap.add_argument("--out", default="../data/clevr_cross_axis_images")
    args = ap.parse_args()
    src_dir = Path(args.clevr_val_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(TRIPLETS_PATH) as f:
        triplets = json.load(f)
    needed = sorted({it["image_path"] for items in triplets.values() for it in items})

    copied, missing = 0, []
    for fname in needed:
        src = src_dir / fname
        if not src.exists():
            missing.append(fname)
            continue
        shutil.copy2(src, out_dir / fname)
        copied += 1

    print(f"copied {copied}/{len(needed)} images -> {out_dir}")
    if missing:
        print(f"MISSING {len(missing)} files (not found in {src_dir}):")
        for m in missing[:10]:
            print(" ", m)
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")


if __name__ == "__main__":
    main()
