"""Regenerates sameaxis_images/ + sameaxis_manifest.json: A, B, C sorted along ONE tested
axis (+ distractor D, forced outside [A,C], never queried). Unlike triplet3ax (A->B/B->C
pure on DIFFERENT axes, r_AC a diagonal mix), everything here is on ONE axis -- r_AC is a
genuine single-axis span.

Reuses triplet3ax's world bounds/object size/overlap guard (`_bounds`, `MIN_DIST_3D`,
`ALL_COMBOS` from generate_triplet3ax.py) -- same renderer, same camera, so a "vertical"
or "closefar" scene here occupies exactly the same world region as triplet3ax's.

Usage: python generate_sameaxis.py [--out-images DIR] [--out-json PATH]
"""
import argparse
import json
import random
from pathlib import Path

from generate_triplet3ax import ALL_COMBOS, MIN_DIST_3D, OBJ_SIZE, _bounds
from render3d import render_multiobject_scene

AXES = ["horizontal", "vertical", "closefar"]
N_PER_AXIS = 60
SEED = {"horizontal": 3042, "vertical": 3142, "closefar": 3242}
X_JITTER = 1.4
X_JITTER_CLOSEFAR = 1.9  # wider -- closefar objects get stratified X lanes, see closefar_lanes()
Y_MID_LO, Y_MID_HI = 4.3, 8.2
MIN_GAP_AXIS = {"horizontal": 1.15, "vertical": 0.55, "closefar": 1.3}  # along the tested axis
D_GAP = 0.3  # min gap from A/C when forcing the distractor outside [A,C]


def desc(shape, color):
    return f"{color} {shape}"


def dist3(p, q):
    return sum((a - b) ** 2 for a, b in zip(p, q)) ** 0.5


def sample_axis_scalars(axis, min_gap, rng):
    """3 sorted, sufficiently-separated scalars along the tested axis for A, B, C."""
    lo, hi = _bounds(axis)
    for _try in range(2000):
        vals = sorted(rng.uniform(lo, hi) for _ in range(3))
        if vals[1] - vals[0] >= min_gap and vals[2] - vals[1] >= min_gap:
            return vals
    raise RuntimeError(f"could not sample separated scalars for axis={axis}")


def world_pos(axis, scalar, rng):
    """Given the tested-axis scalar for one of A/B/C, sample the other 2 (jittered) world
    coords -> returns (x, y, z_base)."""
    if axis == "horizontal":
        return scalar, rng.uniform(Y_MID_LO, Y_MID_HI), 0.0
    if axis == "closefar":
        return rng.uniform(-X_JITTER_CLOSEFAR, X_JITTER_CLOSEFAR), scalar, 0.0
    if axis == "vertical":
        return rng.uniform(-X_JITTER, X_JITTER), rng.uniform(Y_MID_LO, Y_MID_HI), scalar
    raise ValueError(axis)


def sample_distractor(axis, rng, a_scalar, c_scalar):
    """Distractor's tested-axis coordinate is forced OUTSIDE [A,C] -- below A or above C,
    ~50/50 (falls back to whichever side has room)."""
    lo, hi = _bounds(axis)
    room_below = (a_scalar - D_GAP) - lo
    room_above = hi - (c_scalar + D_GAP)
    can_below, can_above = room_below > 0.05, room_above > 0.05
    if can_below and can_above:
        use_below = rng.random() < 0.5
    elif can_below:
        use_below = True
    elif can_above:
        use_below = False
    else:
        raise RuntimeError(f"no room to place distractor outside [A,C] for axis={axis}")
    tested = rng.uniform(lo, a_scalar - D_GAP) if use_below else rng.uniform(c_scalar + D_GAP, hi)

    if axis == "horizontal":
        return tested, rng.uniform(Y_MID_LO, Y_MID_HI), 0.0
    if axis == "closefar":
        return rng.uniform(-X_JITTER_CLOSEFAR, X_JITTER_CLOSEFAR), tested, 0.0
    if axis == "vertical":
        return rng.uniform(-X_JITTER, X_JITTER), rng.uniform(Y_MID_LO, Y_MID_HI), tested
    raise ValueError(axis)


def object_center(x, y, z_base, size):
    """draw_object's cz is the object's BASE, not center -- sphere center = cz+r,
    cube/cylinder/cone span [cz, cz+size]. size/2 approximates the center for all 4
    shapes closely enough for ground-truth purposes."""
    return (x, y, z_base + size / 2)


def closefar_lanes(rng):
    """4 well-separated X lanes + small jitter, shuffled across A/B/C/D -- prevents two
    objects from sharing nearly the same X (which perspective would stack visually)."""
    lanes = [-1.7, -1.7 + 3.4 / 3, -1.7 + 2 * 3.4 / 3, 1.7]
    rng.shuffle(lanes)
    return [lane + rng.uniform(-0.1, 0.1) for lane in lanes]


def build_scene(axis, rng):
    min_gap = MIN_GAP_AXIS[axis]
    for _attempt in range(200):
        scalars = sample_axis_scalars(axis, min_gap, rng)
        combos = rng.sample(ALL_COMBOS, 4)
        lanes = closefar_lanes(rng) if axis == "closefar" else None
        objs = []
        for i, (label, scalar) in enumerate(zip("ABC", scalars)):
            x, y, z_base = world_pos(axis, scalar, rng)
            if lanes is not None:
                x = lanes[i]
            objs.append({"label": label, "shape": combos[i][0], "color": combos[i][1],
                         "x": x, "y": y, "z_base": z_base, "scalar": scalar})
        try:
            dx, dy, dz_base = sample_distractor(axis, rng, scalars[0], scalars[2])
        except RuntimeError:
            continue
        if lanes is not None:
            dx = lanes[3]
        objs.append({"label": "D", "shape": combos[3][0], "color": combos[3][1],
                     "x": dx, "y": dy, "z_base": dz_base, "scalar": None})

        centers = [object_center(o["x"], o["y"], o["z_base"], OBJ_SIZE) for o in objs]
        ok = all(dist3(centers[i], centers[j]) >= MIN_DIST_3D
                 for i in range(4) for j in range(i + 1, 4))
        if ok:
            for o, c in zip(objs, centers):
                o["center"] = c
                o["desc"] = desc(o["shape"], o["color"])
            return objs
    raise RuntimeError(f"could not place non-overlapping scene axis={axis}")


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--out-images", default="../data/sameaxis_images")
    ap_.add_argument("--out-json", default="../data/sameaxis_manifest.json")
    args = ap_.parse_args()
    img_dir = Path(args.out_images)
    img_dir.mkdir(parents=True, exist_ok=True)

    manifest = {}
    for axis in AXES:
        rng = random.Random(SEED[axis])
        scenes_out = []
        for idx in range(N_PER_AXIS):
            objs = build_scene(axis, rng)
            fname = f"{axis}_{idx:03d}.png"
            render_multiobject_scene(objs, OBJ_SIZE, img_dir / fname)
            by_label = {o["label"]: o for o in objs}
            scenes_out.append({
                "image_path": fname, "axis": axis,
                "A": by_label["A"]["desc"], "B": by_label["B"]["desc"],
                "C": by_label["C"]["desc"], "D": by_label["D"]["desc"],
                "objects": {o["label"]: {"desc": o["desc"], "center": list(o["center"])} for o in objs},
            })
            if (idx + 1) % 15 == 0:
                print(f"[{axis}] {idx+1}/{N_PER_AXIS} rendered")
        manifest[axis] = scenes_out

    with open(args.out_json, "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"saved -> {args.out_json}")


if __name__ == "__main__":
    main()
