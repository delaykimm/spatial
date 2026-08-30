"""Regenerates aug1_images/: two probe objects (shape+color), one at the '+' slot and one
at the '-' slot along a single axis (horizontal/vertical/closefar), rendered twice per
scene -- 'orig' (as assigned) and 'swap' (slots reassigned, the matched-pair
counterpart). 60 scenes/axis, seeded -- verified byte-identical to the original dataset.

vertical is a physically touching stack (bottom object's top exactly meets the top
object's base) rather than two floating objects, so above/below has zero ambiguity --
needs manual draw order (computed_zorder=False) instead of automatic depth-sorting,
since automatic sorting could clip the wrong object at a zero gap regardless of shape.

Usage: python generate_aug1.py [--out DIR]
"""
import argparse
import random
from pathlib import Path

from render3d import COLORS, SHAPES, X_RANGE, Y_FAR, Y_NEAR, draw_object, new_scene_figure, save_scene

SEED = {"horizontal": 3042, "vertical": 4042, "closefar": 5042}
N_SCENES = 60
PROBE_SIZE = 1.0
Y_MID = (Y_NEAR + Y_FAR) / 2
X_LEFT, X_RIGHT = -(X_RANGE - 0.5), (X_RANGE - 0.5)
Z_LOW, Z_HIGH = 0.0, PROBE_SIZE
NEAR_Y, FAR_Y = Y_NEAR + 0.4, Y_FAR - 0.4


def build_scene(scene_id, axis, rng):
    shape_a, shape_b = rng.sample(SHAPES, 2)
    color_a, color_b = rng.sample(COLORS, 2)
    a_slot = rng.choice(["plus", "minus"])
    b_slot = "minus" if a_slot == "plus" else "plus"
    return {
        "scene_id": scene_id, "axis": axis,
        "A": {"shape": shape_a, "color": color_a, "slot": a_slot},
        "B": {"shape": shape_b, "color": color_b, "slot": b_slot},
    }


def _pos(axis, slot):
    if axis == "horizontal":
        return (X_RIGHT if slot == "plus" else X_LEFT), Y_MID, 0.0
    elif axis == "vertical":
        return 0.0, Y_MID, (Z_HIGH if slot == "plus" else Z_LOW)
    else:  # closefar
        return 0.0, (NEAR_Y if slot == "plus" else FAR_Y), 0.0


def render(scene, swap, out_dir):
    """swap=False: A/B at their own slot. swap=True: reassign which of A/B takes the
    '+' vs '-' slot (matched-pair counterpart)."""
    axis = scene["axis"]
    A, B = dict(scene["A"]), dict(scene["B"])
    a_slot = ("minus" if A["slot"] == "plus" else "plus") if swap else A["slot"]
    b_slot = ("minus" if B["slot"] == "plus" else "plus") if swap else B["slot"]
    ax_, ay_, az_ = _pos(axis, a_slot)
    bx_, by_, bz_ = _pos(axis, b_slot)
    excl = [(ax_, ay_, az_ + PROBE_SIZE / 2, PROBE_SIZE * 0.85),
            (bx_, by_, bz_ + PROBE_SIZE / 2, PROBE_SIZE * 0.85)]
    fig, ax = new_scene_figure(exclude_objects=excl, computed_zorder=(axis != "vertical"))
    a_anchor = b_anchor = axis != "vertical"
    # manual compositing (vertical) draws later artists over earlier ones unconditionally,
    # so the bottom (lower Z) object must be added first regardless of A/B identity.
    if axis == "vertical" and az_ > bz_:
        (ax_, ay_, az_, A), (bx_, by_, bz_, B) = (bx_, by_, bz_, B), (ax_, ay_, az_, A)
    draw_object(ax, A["shape"], A["color"], ax_, ay_, az_, PROBE_SIZE, anchor=a_anchor)
    draw_object(ax, B["shape"], B["color"], bx_, by_, bz_, PROBE_SIZE, anchor=b_anchor)
    tag = "swap" if swap else "orig"
    path = out_dir / f"{axis}_{scene['scene_id']}_{tag}.png"
    save_scene(fig, path)


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--out", default="../data/aug1_images")
    args = ap_.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for axis in ["horizontal", "vertical", "closefar"]:
        rng = random.Random(SEED[axis])
        for i in range(N_SCENES):
            scene = build_scene(f"scene{i:03d}", axis, rng)
            render(scene, False, out_dir)
            render(scene, True, out_dir)
        print(f"[{axis}] {N_SCENES} scenes x2 rendered -> {out_dir}")


if __name__ == "__main__":
    main()
