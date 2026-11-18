"""Refine a recorded path by gradient descent through the spray model.

Each pose gets a bounded dwell (inverse speed, mean 1 so cycle time is unchanged), a
standoff offset and a shift across the spray axis. Spray angles stay as recorded.
"""
import math

import numpy as np
import torch

from src.spray import deposit, film, perpendicular_basis, sample_surface, spray_directions, subsample


def refine(sample, standoff=200.0, cone_half_angle=25.0, steps=100, lr=0.05, w_smooth=10.0,
           max_dwell=2.0, max_standoff=50.0, max_shift=15.0, n_points=8000, n_eval=20000,
           device=None, log_every=25):
    """Returns the corrected trajectory (X, Y, Z, A, B, C, standoff, speed, stroke) and
    the film before and after.

    The fit uses a soft cone edge on one set of surface points; the films are scored
    with the hard edge on another set, so they are not overfit.
    """
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    as_t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
    poses, ids = subsample(sample["strokes"])
    dirs = spray_directions(poses[:, 3:6])
    e1, e2 = perpendicular_basis(dirs)
    xyz_t, dirs_t, e1_t, e2_t = map(as_t, (poses[:, :3], dirs, e1, e2))
    same_stroke = torch.as_tensor((ids[2:] == ids[1:-1]) & (ids[1:-1] == ids[:-2]), device=device)
    raw = torch.zeros(len(poses), 4, device=device, requires_grad=True)

    def path():
        b = torch.tanh(raw)
        dwell = torch.exp(math.log(max_dwell) * b[:, 0])
        aim = xyz_t + max_shift * (b[:, 2:3] * e1_t + b[:, 3:4] * e2_t)
        return aim, standoff + max_standoff * b[:, 1], dwell / dwell.mean(), b

    points, normals = map(as_t, sample_surface(sample["vertices"], sample["faces"], n_points, seed=1))
    opt = torch.optim.Adam([raw], lr=lr)
    for i in range(steps + 1):
        aim, gap, dwell, b = path()
        t = deposit(points, normals, aim - gap[:, None] * dirs_t, dirs_t, cone_half_angle, dwell,
                    edge_width=1.0)
        t = t / t.mean()
        # Evenness, plus a penalty on thin spots
        fit = t.var(correction=0) + 10 * torch.relu(0.5 - t).square().mean()
        if log_every and i % log_every == 0:
            print(f"  step {i:4d}  loss={fit.item():.4f}")
        if i == steps:
            break
        d2 = b[2:] - 2 * b[1:-1] + b[:-2]
        opt.zero_grad()
        (fit + w_smooth * d2[same_stroke].square().mean()).backward()
        opt.step()

    with torch.no_grad():
        aim, gap, dwell = (a.cpu().double().numpy() for a in path()[:3])
    points, normals = sample_surface(sample["vertices"], sample["faces"], n_eval, seed=0)
    return {
        "points": points,
        "normals": normals,
        "before": film(points, normals, poses[:, :3] - standoff * dirs, dirs, cone_half_angle),
        "after": film(points, normals, aim - gap[:, None] * dirs, dirs, cone_half_angle, dwell),
        "trajectory": np.column_stack([aim, poses[:, 3:6], gap, 1 / dwell, ids]),
    }
