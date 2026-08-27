"""Geometric alignment: does the model's raw relation vector (r_AB/r_BC/r_AC) point the
SAME DIRECTION as the REAL physical displacement, on every one of this dataset's axes?

- Method (dual dot-product, not QR -- QR can sign-flip when calibration axes aren't
  exactly orthogonal): model_A(L) = r_pair(L) . unit(own_axis[A](L)); real_A = end2's real
  position along A minus end1's. Cosine per layer vs a K=1000 derangement permutation null.
  (cross_axis_readout_vqa.py asks a simpler question on the same data -- just sign-match
  vs a VQA baseline, not full-direction cosine.)
- Own-axis calibration is sign-corrected (build_own_axes_signed): each leg's +/- sign is
  random, so pooling without correcting lets opposite-signed legs cancel out.
- Real position: CLEVR uses clevr_val_scenes.json's camera-relative right/behind (no
  vertical); triplet3ax replays the same seeded RNG used at scene-build time (no
  re-rendering).

Usage:
    python cross_axis_alignment.py --dataset clevr
    python cross_axis_alignment.py --dataset triplet3ax --model llava
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse

import numpy as np

import triplet_pipeline as cxp
import cross_axis_analysis as caa
import axis_pipeline as ap

_CFG = ap.CFG["cross_axis_alignment"]
K = _CFG["k_permtest"]
SEED = _CFG["seed"]
PAIRS = cxp.PAIRS
AXES_BY_DATASET = cxp.AXES_BY_DATASET
REPORT_LAYERS_FRAC = _CFG["report_layers_frac"]  # ~8 evenly spread layers, any model
MODEL_DISPLAY = _CFG["model_display"]

report_layers = ap.report_layers_frac  # canonical version now lives in axis_pipeline.py


# =============================================================================
# alignment: per-pair, all-axes-at-once cosine + permutation test
# =============================================================================
def run_alignment(dataset, model):
    ap.set_model(model)
    axes = AXES_BY_DATASET[dataset]
    triplets = caa.load_all_triplets(dataset)
    N = len(triplets)
    num_l = triplets[0]["r_AB"].shape[0]
    groups = caa.group_split(N)

    real_pos = cxp.real_positions(dataset)
    own_axes = cxp.build_own_axes_signed(triplets, groups, real_pos)
    print(f"[{model}][{dataset}] N={N} cross-axis triplets, axes={axes}, num_layers={num_l}")

    r_field = {"AB": "r_AB", "BC": "r_BC", "AC": "r_AC"}
    rng = np.random.default_rng(SEED)
    results = {}
    for pair in PAIRS:
        model_allaxes = np.zeros((N, num_l, len(axes)))
        real_allaxes = np.zeros((N, len(axes)))
        for i, (t, g) in enumerate(zip(triplets, groups)):
            other = 2 if g == 1 else 1
            pos = real_pos[(t["config"], t["idx"])]
            real_allaxes[i] = cxp.real_displacement_vec(pos, pair, axes)
            r_vec = t[r_field[pair]]  # (num_l, d)
            for ai, ax in enumerate(axes):
                axis_unit = caa.unit(own_axes[other][ax])  # (num_l, d)
                model_allaxes[i, :, ai] = np.sum(r_vec * axis_unit, axis=-1)

        model_unit = caa.unit(model_allaxes, axis=-1)  # (N,num_l,k)
        real_unit = caa.unit(real_allaxes, axis=-1)     # (N,k)
        test = caa.cosine_permtest(model_unit, real_unit, rng, K)

        results[pair] = dict(model_allaxes=model_allaxes, real_allaxes=real_allaxes, **test)
        print(f"  [{pair}] done")

    return results, axes, num_l


def print_summary(results, axes, dataset, model, num_l):
    layers = report_layers(num_l)
    print(f"\n{'='*100}\n[{model}][{dataset}] GROUNDING: cos(model all-axes vector, real all-axes "
          f"displacement), axes={axes}\n{'='*100}")
    header = f"{'layer':>5} " + " ".join(f"{p:>14}" for p in PAIRS)
    print(header + "   (cos / z)")
    for L in layers:
        row = f"{L:5d} " + " ".join(
            f"{results[p]['obs_mean'][L]:6.3f}/{results[p]['z'][L]:6.1f}" for p in PAIRS)
        print(row)


# =============================================================================
# joint compositionality + grounding + magnitude-fidelity summary (mirrors ours/binding's
# plot_composition_grounding_joint.py) -- own axis only, no external-dataset dependency
# =============================================================================
def run_composition_grounding_joint(dataset, model):
    """Compositionality (cos(r_AB+r_BC, r_AC), no projection) + magnitude fidelity
    (1 - norm_resid), same "raw" battery as cross_axis_analysis.run_permtest_battery, plus
    AC-leg grounding reused from a cached alignment npz (run_alignment first). No new GPU
    work."""
    align_path = f"{ap.RESULTS}/alignment/{dataset}_alignment_{model}.npz"
    if not os.path.exists(align_path):
        raise FileNotFoundError(f"{align_path} not found -- run cross_axis_alignment.py --dataset {dataset} "
                                 f"--model {model} first")
    align_d = np.load(align_path, allow_pickle=True)
    ground_mean = align_d["AC_obs_mean"]
    num_l = len(ground_mean)

    triplets = caa.load_all_triplets(dataset)
    pred_raw = np.stack([t["r_AB"] + t["r_BC"] for t in triplets])
    targ_raw = np.stack([t["r_AC"] for t in triplets])
    rng = np.random.default_rng(caa.SEED)
    comp = caa.composition_battery(pred_raw, targ_raw, rng, num_l)
    comp_mean = comp["cos"].mean(axis=0)
    magfid_mean = 1 - comp["resid"].mean(axis=0)

    return dict(comp_mean=comp_mean, ground_mean=ground_mean, magfid_mean=magfid_mean, num_l=num_l)


def draw_composition_grounding_joint_plot(dataset):
    """Mirrors ours/binding's plot_composition_grounding_joint.py: one panel per model
    with a cached {dataset}_alignment_{model}.npz on disk (run_composition_grounding_joint
    recomputes compositionality/magnitude-fidelity fresh each call -- cheap, CPU-only,
    reuses the already-extracted shard)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURFACE, INK, SECONDARY, MUTED, GRID = "#ffffff", "#1a1a19", "#52514e", "#898781", "#e5e4df"
    COMP_COLOR, GROUND_COLOR, MAGFID_COLOR, PEAK_COLOR = "#2a78d6", "#1baf7a", "#9b6bd6", "#eb6834"

    per_model = {}
    for model in ap.MODELS:
        if os.path.exists(f"{ap.RESULTS}/alignment/{dataset}_alignment_{model}.npz"):
            per_model[model] = run_composition_grounding_joint(dataset, model)
    if not per_model:
        print(f"[{dataset}] no cached alignment npz found for any model, skipping joint plot")
        return

    fig, axes_plt = plt.subplots(1, len(per_model), figsize=(6.5 * len(per_model), 6.0), dpi=150,
                                  facecolor=SURFACE, sharey=True, squeeze=False)
    axes_plt = axes_plt[0]
    fig.suptitle("Where should relational/spatial info be read out? Compositionality, grounding "
                 "& magnitude fidelity, by layer", color=INK, fontsize=15.5, fontweight="bold",
                 x=0.03, ha="left", y=1.03)
    summary_rows = []
    for col, (model, d) in enumerate(per_model.items()):
        num_l = d["num_l"]
        depth = np.arange(num_l) / (num_l - 1)
        joint = np.minimum(d["comp_mean"], d["ground_mean"])
        best_L = int(np.argmax(joint))
        summary_rows.append((model, num_l, best_L, depth[best_L], d["comp_mean"][best_L],
                              d["ground_mean"][best_L], d["magfid_mean"][best_L]))

        ax = axes_plt[col]
        ax.set_facecolor(SURFACE)
        ax.plot(depth, d["comp_mean"], color=COMP_COLOR, linewidth=2.2, label="additive compositionality",
                 solid_capstyle="round", zorder=3)
        ax.plot(depth, d["ground_mean"], color=GROUND_COLOR, linewidth=2.2, label="geometric grounding",
                 solid_capstyle="round", zorder=3)
        ax.plot(depth, d["magfid_mean"], color=MAGFID_COLOR, linewidth=2.0, linestyle=(0, (5, 2)),
                 label="magnitude fidelity", solid_capstyle="round", zorder=3)
        ax.axvline(depth[best_L], color=PEAK_COLOR, linewidth=1.3, linestyle=(0, (3, 2)), alpha=0.85, zorder=2)
        ax.scatter([depth[best_L]], [joint[best_L]], color=PEAK_COLOR, s=90, marker="*", zorder=5,
                   edgecolors=SURFACE, linewidths=0.8)
        ax.annotate(f"{depth[best_L]*100:.0f}%\n(L{best_L})", xy=(depth[best_L], joint[best_L]),
                    xytext=(depth[best_L] + 0.05 if depth[best_L] < 0.8 else depth[best_L] - 0.30, 0.18),
                    color=PEAK_COLOR, fontsize=10, fontweight="bold", ha="left")

        ax.axhline(0, color=MUTED, linewidth=1, linestyle=(0, (1, 2)), alpha=0.6, zorder=1)
        ax.set_title(f"{MODEL_DISPLAY.get(model, model)}  ({num_l} layers)", color=INK, fontsize=12.5,
                     fontweight="bold", pad=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.15, 1.05)
        ax.set_xlabel("relative depth (layer / max layer)", color=INK, fontsize=10.5)
        ax.xaxis.set_major_formatter(lambda x, pos: f"{x*100:.0f}%")
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=SECONDARY, length=0, labelsize=9.5)
        if col == 0:
            ax.set_ylabel("cosine similarity", color=INK, fontsize=11)
        else:
            ax.tick_params(labelleft=False)

    handles = [plt.Line2D([0], [0], color=COMP_COLOR, linewidth=2.2),
               plt.Line2D([0], [0], color=GROUND_COLOR, linewidth=2.2),
               plt.Line2D([0], [0], color=MAGFID_COLOR, linewidth=2.0, linestyle=(0, (5, 2))),
               plt.Line2D([0], [0], color=PEAK_COLOR, linewidth=0, marker="*", markersize=11)]
    fig.legend(handles, ["additive compositionality", "geometric grounding",
                          "magnitude fidelity (1−norm_resid)", "joint peak"],
               loc="lower center", ncols=4, frameon=False, fontsize=10.5, labelcolor=SECONDARY,
               bbox_to_anchor=(0.5, -0.02), handlelength=1.8, handletextpad=0.6, columnspacing=1.6)

    fig.subplots_adjust(wspace=0.05, top=0.83, bottom=0.20, left=0.055, right=0.98)
    out_path = f"{ap.PLOTS}/alignment/{dataset}_composition_grounding_joint.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor=SURFACE)
    print(f"saved -> {out_path}")

    print(f"\n{'model':<10} {'layer':>6} {'depth':>7} {'comp':>7} {'ground':>7} {'magfid':>7}")
    for model, num_l, L, depth_L, c, g, m in summary_rows:
        print(f"{model:<10} {L:6d} {depth_L*100:6.0f}% {c:7.3f} {g:7.3f} {m:7.3f}")


# =============================================================================
# dataset-axis alignment: calibrate against each of the 4 external datasets' axes instead
# of this dataset's own -- does alignment hold on a totally different image distribution?
# =============================================================================
EXT_AXIS_FLIP = set(_CFG["ext_axis_flip"])  # axes needing a sign flip when borrowed from an external dataset


def load_external_axis_units(ext_ds, axes):
    """{axis: (num_l,d) unit vector}, sign-corrected for a naming mismatch: external
    datasets define closefar+ as "front" (toward camera), CLEVR/triplet3ax define it as
    +=far -- opposite polarity, so flip closefar here (verified empirically via cos(own,
    external))."""
    out = {}
    for ax in axes:
        v = caa.unit(caa.external_axis_vector(ext_ds, ax))
        out[ax] = -v if ax in EXT_AXIS_FLIP else v
    return out


def _precompute_pair_data(dataset):
    """{pair: (r_full (N,num_l,d), real_unit (N,k))}, shared across every axis source
    (own + all 4 external) so real-world ground truth / r_pair stacking is done once."""
    triplets = caa.load_all_triplets(dataset)
    axes = AXES_BY_DATASET[dataset]
    real_pos = cxp.real_positions(dataset)
    r_field = {"AB": "r_AB", "BC": "r_BC", "AC": "r_AC"}
    out = {}
    for pair in PAIRS:
        r_full = np.stack([t[r_field[pair]] for t in triplets])
        real_full = np.stack([cxp.real_displacement_vec(real_pos[(t["config"], t["idx"])], pair, axes)
                               for t in triplets])
        out[pair] = (r_full, caa.unit(real_full))
    return out, len(triplets), axes


def run_alignment_datasetaxis(dataset, model):
    ap.set_model(model)
    pair_data, N, axes = _precompute_pair_data(dataset)
    num_l = pair_data["AB"][0].shape[1]
    print(f"[{model}][{dataset}] dataset-axis alignment: N={N}, axes={axes}, num_layers={num_l}")

    rng = np.random.default_rng(SEED)
    all_out = {}
    for ext_ds in caa.EXTERNAL_DATASETS:
        axis_units = load_external_axis_units(ext_ds, axes)
        pair_results = {}
        for pair in PAIRS:
            r_full, real_unit = pair_data[pair]
            model_allaxes = np.stack([np.einsum("nld,ld->nl", r_full, axis_units[ax]) for ax in axes], axis=-1)
            model_unit = caa.unit(model_allaxes, axis=-1)
            pair_results[pair] = caa.cosine_permtest(model_unit, real_unit, rng, K)

        all_out[ext_ds] = pair_results
        layers = report_layers(num_l)
        print(f"  [{ext_ds}][AC] " + " ".join(f"L{L}={pair_results['AC']['obs_mean'][L]:.3f}"
                                                f"(z={pair_results['AC']['z'][L]:.1f})" for L in layers))
    return all_out, num_l


def print_datasetaxis_summary(all_out, own_results, dataset, model, num_l):
    """AC-leg alignment cos, by axis source (own + 4 external) -- mirrors
    clevr_geometric_grounding_datasetaxis.py's summary table."""
    layers = report_layers(num_l)
    print(f"\n{'='*100}\n[{model}][{dataset}] SUMMARY: AC-leg alignment cos, by axis source\n{'='*100}")
    header = f"{'layer':>5} {'own axis':>10} " + " ".join(f"{d:>14}" for d in caa.EXTERNAL_DATASETS)
    print(header)
    for L in layers:
        row = f"{L:5d} {own_results['AC']['obs_mean'][L]:10.3f} " + " ".join(
            f"{all_out[d]['AC']['obs_mean'][L]:14.3f}" for d in caa.EXTERNAL_DATASETS)
        print(row)


def draw_datasetaxis_comparison_plot(dataset):
    """AC leg only, one panel per model with both a cached alignment npz (own axis) and a
    datasetaxis npz (4 external axes) on disk -- reads only, never recomputes. X-axis is
    relative depth (layer / max_layer) so models with different layer counts overlay."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURFACE, INK, SECONDARY, MUTED, GRID = "#ffffff", "#1a1a19", "#52514e", "#898781", "#e5e4df"
    COLORS = {"own axis": "#eb6834", "whatsup": "#1baf7a", "spatialtunnel": "#eda100",
              "aug1": "#e87ba4", "aug2": "#008300"}
    ORDER = ["own axis"] + caa.EXTERNAL_DATASETS

    per_model = {}
    for model in ap.MODELS:
        own_path = f"{ap.RESULTS}/alignment/{dataset}_alignment_{model}.npz"
        ds_path = f"{ap.RESULTS}/alignment/{dataset}_alignment_datasetaxis_{model}.npz"
        if os.path.exists(own_path) and os.path.exists(ds_path):
            per_model[model] = (np.load(own_path, allow_pickle=True), np.load(ds_path, allow_pickle=True))
    if not per_model:
        print(f"[{dataset}] no cached own+datasetaxis alignment npz found for any model, skipping plot")
        return

    fig, axes_plt = plt.subplots(1, len(per_model), figsize=(6.5 * len(per_model), 5.6), dpi=150,
                                  facecolor=SURFACE, sharey=True, squeeze=False)
    axes_plt = axes_plt[0]
    fig.suptitle(f"{dataset}: geometric alignment by axis calibration source (AC leg)",
                 color=INK, fontsize=15.5, fontweight="bold", x=0.03, ha="left", y=1.02)

    for col, (model, (own_d, ds_d)) in enumerate(per_model.items()):
        ax = axes_plt[col]
        ax.set_facecolor(SURFACE)
        num_l = len(own_d["AC_obs_mean"])
        depth = np.arange(num_l) / (num_l - 1)
        for name in ORDER:
            cos_mean = own_d["AC_obs_mean"] if name == "own axis" else ds_d[f"{name}_AC_obs_mean"]
            ax.plot(depth, cos_mean, color=COLORS[name], linewidth=2.2,
                     label=name, solid_capstyle="round", zorder=3)
        ax.axhline(0, color=MUTED, linewidth=1, linestyle=(0, (1, 2)), alpha=0.7, zorder=1)
        ax.set_title(f"{MODEL_DISPLAY.get(model, model)}  ({num_l} layers)", color=INK, fontsize=12.5,
                     fontweight="bold", pad=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.15, 1.02)
        ax.set_xlabel("relative depth (layer / max layer)", color=INK, fontsize=10.5)
        ax.xaxis.set_major_formatter(lambda x, pos: f"{x*100:.0f}%")
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=SECONDARY, length=0, labelsize=9.5)
        if col == 0:
            ax.set_ylabel("cos(model dir, real physical dir)", color=INK, fontsize=11)
        else:
            ax.tick_params(labelleft=False)

    handles = [plt.Line2D([0], [0], color=COLORS[n], linewidth=2.2) for n in ORDER]
    fig.legend(handles, ORDER, loc="lower center", ncols=len(ORDER),
               frameon=False, fontsize=10.5, labelcolor=SECONDARY, bbox_to_anchor=(0.5, -0.02),
               handlelength=1.8, handletextpad=0.6, columnspacing=1.6)

    fig.subplots_adjust(wspace=0.05, top=0.82, bottom=0.22, left=0.055, right=0.98)
    out_path = f"{ap.PLOTS}/alignment/{dataset}_alignment_datasetaxis_comparison.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor=SURFACE)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--dataset", required=True, choices=cxp.DATASETS)
    cli.add_argument("--model", choices=ap.MODELS, default=ap.DEFAULT_MODEL)
    args = cli.parse_args()

    results, axes, num_l = run_alignment(args.dataset, args.model)
    print_summary(results, axes, args.dataset, args.model, num_l)

    out_path = f"{ap.RESULTS}/alignment/{args.dataset}_alignment_{args.model}.npz"
    flat = {"axes": np.array(axes)}
    for pair, r in results.items():
        for k, v in r.items():
            flat[f"{pair}_{k}"] = v
    np.savez(out_path, **flat)
    print(f"\nsaved -> {out_path}")

    ds_results, _ = run_alignment_datasetaxis(args.dataset, args.model)
    print_datasetaxis_summary(ds_results, results, args.dataset, args.model, num_l)

    ds_out_path = f"{ap.RESULTS}/alignment/{args.dataset}_alignment_datasetaxis_{args.model}.npz"
    ds_flat = {f"{ext_ds}_{pair}_{k}": v for ext_ds, pr in ds_results.items()
               for pair, r in pr.items() for k, v in r.items()}
    np.savez(ds_out_path, **ds_flat)
    print(f"\nsaved -> {ds_out_path}")

    draw_datasetaxis_comparison_plot(args.dataset)
