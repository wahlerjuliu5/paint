"""The experiments behind the figures in outputs/."""
import csv

import matplotlib.pyplot as plt
import numpy as np

from src.data import OUTPUT_DIR, list_samples, load_sample
from src.metrics import cv, in_tolerance, paint_coverage, pc_bar, side_profiles, stripe_cv
from src.refine import refine
from src.spray import film, gun_poses, perpendicular_basis, sample_surface, spray_directions, subsample

STANDOFFS = {120.0: "tab:red", 200.0: "tab:blue"}  # the papers' stated distance, and ours
METRICS = {"cv": "Thickness CV", "stripe_cv": "Stripe CV along Z (side faces)",
           "in_tolerance": "Surface within ±20% of mean", "pc_bar": "Papers' PC bar / mean"}


def summarize(points, normals, thickness):
    return {"cv": cv(thickness), "stripe_cv": stripe_cv(points, normals, thickness),
            "in_tolerance": in_tolerance(thickness),
            "pc_bar": pc_bar(thickness) / thickness.mean()}


def evaluate_recorded(index, settings, n_points=8000, step=5):
    """Metrics of one part's recorded path for each (standoff, cone half-angle)."""
    sample = load_sample(index)
    points, normals = sample_surface(sample["vertices"], sample["faces"], n_points)
    return [{"name": sample["name"], "standoff": standoff, "cone": cone,
             **summarize(points, normals, film(points, normals,
                                               *gun_poses(sample["strokes"], standoff, step), cone))}
            for standoff, cone in settings]


def smooth_noise(rng, n, corr=40):
    """Unit-variance noise, correlated over about corr samples."""
    white = rng.standard_normal(n + corr - 1)
    return np.convolve(white, np.ones(corr) / corr, mode="valid") * np.sqrt(corr)


def sloppy_path(strokes, standoff, jitter=30.0, dwell_spread=0.5, seed=0):
    """The recorded path with the aim wandering across the spray axis and the speed
    drifting, smoothly along each stroke. Mean dwell stays 1, so it sprays as much paint."""
    rng = np.random.default_rng(seed)
    poses, ids = subsample(strokes)
    dirs = spray_directions(poses[:, 3:6])
    e1, e2 = perpendicular_basis(dirs)
    noise = np.zeros((len(poses), 3))
    for k in np.unique(ids):
        m = ids == k
        noise[m] = np.stack([smooth_noise(rng, m.sum()) for _ in range(3)], axis=1)
    aim = poses[:, :3] + jitter * (noise[:, :1] * e1 + noise[:, 1:2] * e2)
    dwell = np.exp(dwell_spread * noise[:, 2])
    return aim - standoff * dirs, dirs, dwell / dwell.mean()


def flow_to_pass(thickness, reference, target=0.99):
    """Smallest paint-flow factor at which thickness reaches the target PC."""
    bar = pc_bar(reference)
    worst_allowed = np.percentile(thickness[reference >= bar], 100 * (1 - target))
    return max(1.0, 1.0001 * bar / max(worst_allowed, 1e-12))


def plot_films(title, points, normals, films):
    """Per film: thickness map, Z profile on the side faces, and the distribution against
    the papers' PC bar. Thickness is relative to the first film's mean."""
    reference = next(iter(films.values()))
    unit = reference.mean()
    top = max(np.percentile(t, 99.5) for t in films.values()) / unit
    fig = plt.figure(figsize=(16, 4.5 * len(films)))
    for row, (label, t) in enumerate(films.items()):
        ax = fig.add_subplot(len(films), 3, 3 * row + 1, projection="3d")
        sc = ax.scatter(*points.T, c=t / unit, s=2, cmap="viridis", vmin=0,
                        vmax=np.percentile(t, 99) / unit)
        fig.colorbar(sc, ax=ax, shrink=0.6)
        ax.set_box_aspect(np.ptp(points, axis=0))
        ax.set_title(f"{label}: CV {cv(t):.2f}, paper PC {paint_coverage(t, reference):.1%}")

        prof = fig.add_subplot(len(films), 3, 3 * row + 2)
        for face, (z, p) in side_profiles(points, normals, t / unit).items():
            prof.plot(z, p, label=face)
        prof.set_xlabel("Z (mm)")
        prof.set_ylabel("mean thickness")
        prof.legend(fontsize=8)

        hist = fig.add_subplot(len(films), 3, 3 * row + 3)
        hist.hist(t / unit, bins=150, range=(0, top))
        hist.axvline(pc_bar(reference) / unit, color="k", ls="--", label="papers' PC bar")
        hist.set_xlabel("thickness")
        hist.legend()
    fig.suptitle(f"{title} (thickness relative to the {next(iter(films))} mean)")
    fig.tight_layout()
    return fig


def _save(fig, rows, name):
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=150)
    if rows:
        with open(OUTPUT_DIR / f"{name}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"wrote {OUTPUT_DIR / name}.png")


def part(index, standoff=200.0, cone=25.0, refined=False):
    """Print one part's metrics and plot its film, optionally before and after refining."""
    sample = load_sample(index)
    if refined:
        result = refine(sample, standoff, cone)
        points, normals = result["points"], result["normals"]
        films = {"recorded": result["before"], "refined": result["after"]}
        path = OUTPUT_DIR / f"{sample['name']}_refined_{standoff:.0f}mm.txt"
        OUTPUT_DIR.mkdir(exist_ok=True)
        np.savetxt(path, result["trajectory"], delimiter=";", fmt="%.4f", comments="",
                   header="X;Y;Z;A;B;C;standoff;speed;strokeId")
        print(f"wrote {path}")
    else:
        points, normals = sample_surface(sample["vertices"], sample["faces"])
        films = {"recorded": film(points, normals, *gun_poses(sample["strokes"], standoff), cone)}
    for label, t in films.items():
        print(f"{label:>9}: " + "  ".join(f"{k}={v:.3f}" for k, v in summarize(points, normals, t).items()))
    plot_films(f"{sample['name']}, {standoff:.0f} mm", points, normals, films)


def stats():
    """Recorded paths of every cuboid at both standoffs."""
    settings = [(standoff, 25.0) for standoff in STANDOFFS]
    rows = []
    for i in range(len(list_samples())):
        rows += evaluate_recorded(i, settings, n_points=5000, step=10)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1} parts", flush=True)
    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.4 * len(METRICS), 3.8))
    for ax, (key, title) in zip(axes, METRICS.items()):
        for standoff, color in STANDOFFS.items():
            values = [r[key] for r in rows if r["standoff"] == standoff]
            ax.hist(values, bins=40, color=color, alpha=0.6, label=f"{standoff:.0f} mm")
            print(f"{title}, {standoff:.0f} mm: median {np.median(values):.3f}")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle(f"Recorded paths of {len(rows) // 2} cuboids, 25° cone")
    fig.tight_layout()
    _save(fig, rows, "stats")


def sweep(n_parts=8):
    """Recorded paths under a grid of gun assumptions; the papers give no cone angle."""
    cones, standoffs = [15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0], [120.0, 160.0, 200.0, 250.0]
    settings = [(s, c) for s in standoffs for c in cones]
    rows = [r for i in np.linspace(0, len(list_samples()) - 1, n_parts).astype(int)
            for r in evaluate_recorded(i, settings)]
    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.4 * len(METRICS), 4.2))
    for ax, (key, title) in zip(axes, METRICS.items()):
        grid = np.array([[np.mean([r[key] for r in rows if (r["standoff"], r["cone"]) == (s, c)])
                          for c in cones] for s in standoffs])
        im = ax.imshow(grid, cmap="viridis", aspect="auto", origin="lower")
        for (i, j), value in np.ndenumerate(grid):
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if im.norm(value) < 0.5 else "black")
        ax.set_xticks(range(len(cones)), [f"{c:.0f}" for c in cones])
        ax.set_yticks(range(len(standoffs)), [f"{s:.0f}" for s in standoffs])
        ax.set_xlabel("cone half-angle (°)")
        ax.set_ylabel("standoff (mm)")
        ax.set_title(title, fontsize=10)
    fig.suptitle(f"Recorded paths under different gun assumptions, mean of {n_parts} parts")
    fig.tight_layout()
    _save(fig, rows, "sweep")


def challenge(index=0):
    """A sloppy path, with the paint flow raised until it passes the papers' PC."""
    sample = load_sample(index)
    points, normals = sample_surface(sample["vertices"], sample["faces"])
    for standoff in STANDOFFS:
        truth = film(points, normals, *gun_poses(sample["strokes"], standoff))
        guns, dirs, dwell = sloppy_path(sample["strokes"], standoff)
        sloppy = film(points, normals, guns, dirs, weights=dwell)
        flow = flow_to_pass(sloppy, truth)
        print(f"{sample['name']}, {standoff:.0f} mm: sloppy path scores "
              f"{paint_coverage(sloppy, truth):.1%} PC at equal paint and needs {flow:.2f}x the "
              f"paint for 99%, leaving {(flow * sloppy > 2 * truth.mean()).mean():.0%} of the "
              f"surface over twice the recorded mean")
        fig = plot_films(f"{sample['name']}, {standoff:.0f} mm, synthetic sloppy path",
                         points, normals, {"recorded": truth, f"sloppy x{flow:.1f} paint": flow * sloppy})
        _save(fig, None, f"challenge_{standoff:.0f}mm")


def refinement(n_parts=10):
    """Recorded against refined paths for parts across the dataset."""
    rows = []
    for i in np.linspace(0, len(list_samples()) - 1, n_parts).astype(int):
        sample = load_sample(i)
        for standoff in STANDOFFS:
            r = refine(sample, standoff, log_every=0)
            row = {"name": sample["name"], "standoff": standoff}
            for stage in ("before", "after"):
                row |= {f"{k}_{stage}": v for k, v in
                        summarize(r["points"], r["normals"], r[stage]).items()}
            rows.append(row | {"pc": paint_coverage(r["after"], r["before"]),
                               "paint": r["after"].mean() / r["before"].mean(),
                               "mean_standoff": r["trajectory"][:, 6].mean()})
        print(f"  {sample['name']} done", flush=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, key in zip(axes, ["cv", "stripe_cv", "in_tolerance"]):
        for standoff, color in STANDOFFS.items():
            sub = [r for r in rows if r["standoff"] == standoff]
            ax.scatter([r[f"{key}_before"] for r in sub], [r[f"{key}_after"] for r in sub],
                       color=color, label=f"{standoff:.0f} mm")
        ax.axline((0, 0), slope=1, color="k", ls="--", lw=1)
        ax.set_xlabel("recorded")
        ax.set_ylabel("refined")
        ax.set_title(METRICS[key], fontsize=10)
        ax.legend(fontsize=8)
    for standoff in STANDOFFS:
        sub = [r for r in rows if r["standoff"] == standoff]
        median = lambda key: np.median([r[key] for r in sub])
        print(f"{standoff:.0f} mm: within ±20% {median('in_tolerance_before'):.0%} -> "
              f"{median('in_tolerance_after'):.0%}, paper PC {median('pc'):.1%}, "
              f"paint {median('paint'):.2f}x, mean standoff {median('mean_standoff'):.0f} mm")
    fig.suptitle(f"Refinement of {n_parts} parts (dashed line: no change)")
    fig.tight_layout()
    _save(fig, rows, "refinement")


SECTIONS = {"stats": stats, "sweep": sweep, "challenge": challenge, "refinement": refinement}
