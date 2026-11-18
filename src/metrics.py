"""Film metrics on area-uniform surface samples."""
import numpy as np

SIDES = {"+X": (0, 1), "-X": (0, -1), "+Y": (1, 1), "-Y": (1, -1)}


def cv(thickness):
    return thickness.std() / thickness.mean()


def in_tolerance(thickness, tol=0.2):
    """Share of surface within +-tol of the mean thickness."""
    return (np.abs(thickness / thickness.mean() - 1) <= tol).mean()


def pc_bar(reference, percentile=10):
    """The papers' coverage bar: a low percentile of the non-zero reference film."""
    return np.percentile(reference[reference > 0], percentile)


def paint_coverage(thickness, reference):
    """Paint Coverage as in PaintNet and MaskPlanner: the share of points the reference
    covers that thickness also lifts over the bar. Any amount above the bar counts."""
    bar = pc_bar(reference)
    return (thickness[reference >= bar] >= bar).mean()


def side_profiles(points, normals, thickness, bins=40):
    """Mean thickness binned along Z on each side face, as (bin centers, means)."""
    out = {}
    for face, (axis, sign) in SIDES.items():
        on_face = sign * normals[:, axis] >= 0.95
        z, t = points[on_face, 2], thickness[on_face]
        edges = np.linspace(z.min(), z.max(), bins + 1)
        idx = np.clip(np.digitize(z, edges) - 1, 0, bins - 1)
        filled = np.unique(idx)
        out[face] = ((edges[filled] + edges[filled + 1]) / 2,
                     np.array([t[idx == b].mean() for b in filled]))
    return out


def stripe_cv(points, normals, thickness):
    """Strength of the stripes the pass spacing leaves on the side faces: CV of the Z
    profile, averaged over the four faces."""
    return np.mean([cv(p) for _, p in side_profiles(points, normals, thickness).values()])
