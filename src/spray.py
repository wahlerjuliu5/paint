"""Spray model: surface samples, gun poses and paint deposition."""
import math

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint


def sample_surface(vertices, faces, n_points=20000, seed=0):
    """Area-uniform surface points [P, 3] and their outward normals [P, 3]."""
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(cross, axis=1) / 2
    normals = cross / np.maximum(2 * area[:, None], 1e-12)
    # A closed mesh wound outward has positive signed volume
    if np.einsum("ij,ij->i", tri[:, 0], cross).sum() < 0:
        normals = -normals

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(faces), size=n_points, p=area / area.sum())
    u, v = rng.random((2, n_points))
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    t = tri[idx]
    points = t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])
    return points, normals[idx]


def spray_directions(abc):
    """Unit spray axes from A, B, C in degrees, in the PaintNet convention: +X rotated
    by scipy's from_euler("yzx", [B, C, A])."""
    a, b, c = np.radians(abc).T
    x, y, z = np.cos(b) * np.cos(c), np.cos(b) * np.sin(c), -np.sin(b)
    return np.stack([x, np.cos(a) * y - np.sin(a) * z, np.sin(a) * y + np.cos(a) * z], axis=1)


def perpendicular_basis(dirs):
    """Two unit vectors per direction, orthogonal to it and to each other."""
    helper = np.where(np.abs(dirs[:, 2:]) > 0.9, [[1.0, 0, 0]], [[0, 0, 1.0]])
    e1 = np.cross(dirs, helper)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    return e1, np.cross(dirs, e1)


def subsample(strokes, step=5):
    """Every step-th pose of each stroke and its stroke index. Poses are recorded at
    equal time intervals, so each kept pose stands for an equal dwell time."""
    kept = [s[::step] for s in strokes]
    return np.vstack(kept), np.concatenate([np.full(len(s), i) for i, s in enumerate(kept)])


def gun_poses(strokes, standoff=200.0, step=5):
    """Gun positions and spray axes. Recorded positions are aim points on the surface;
    the gun sits standoff mm behind them."""
    poses = subsample(strokes, step)[0]
    dirs = spray_directions(poses[:, 3:6])
    return poses[:, :3] - standoff * dirs, dirs


def _flux(points, normals, guns, dirs, weights, half_angle, edge_width):
    r = points[None] - guns[:, None]
    h = r.norm(dim=2)
    r_hat = r / h[..., None]
    d = dirs[:, None].expand_as(r_hat)
    # atan2 rather than arccos, whose gradient is infinite on the spray axis
    sin_theta = torch.linalg.cross(r_hat, d, dim=2).square().sum(2).add(1e-12).sqrt()
    theta = torch.atan2(sin_theta, (r_hat * d).sum(2))
    cos_beta = -(r_hat * normals[None]).sum(2)
    flux = torch.exp(-0.5 * (theta / (half_angle / 2)) ** 2) * cos_beta / h**2
    if edge_width > 0:
        flux = flux * torch.sigmoid((half_angle - theta) / edge_width)
        lit = cos_beta > 0
    else:
        lit = (theta <= half_angle) & (cos_beta > 0)
    return (torch.where(lit, flux, 0.0) * weights[:, None]).sum(0)


def deposit(points, normals, guns, dirs, cone_half_angle=25.0, weights=None, edge_width=0.0,
            chunk=128):
    """Paint thickness at each surface point, differentiable in every input.

    Each gun pose adds f(theta) cos(beta) / h^2: f is a Gaussian in the angle off the
    spray axis (sigma = half angle / 2, cut at the half angle), beta the incidence angle,
    h the distance. Back-facing points get nothing, which is exact only for convex
    parts. weights are relative dwell times. A small edge_width (degrees) softens the
    cone edge so gradients see paint entering and leaving it.
    """
    like = {"dtype": points.dtype, "device": points.device}
    half_angle = torch.deg2rad(torch.as_tensor(cone_half_angle, **like))
    if weights is None:
        weights = torch.ones(len(guns), **like)
    needs_grad = torch.is_grad_enabled() and any(
        t.requires_grad for t in (points, normals, guns, dirs, weights, half_angle))
    thickness = torch.zeros(len(points), **like)
    for i in range(0, len(guns), chunk):
        args = (points, normals, guns[i:i + chunk], dirs[i:i + chunk], weights[i:i + chunk],
                half_angle, math.radians(edge_width))
        part = checkpoint(_flux, *args, use_reentrant=False) if needs_grad else _flux(*args)
        thickness = thickness + part
    return thickness


def film(points, normals, guns, dirs, cone_half_angle=25.0, weights=None):
    """deposit on NumPy arrays, in float64 and without gradients."""
    as_t = lambda a: None if a is None else torch.as_tensor(a, dtype=torch.float64)
    with torch.no_grad():
        return deposit(*map(as_t, (points, normals, guns, dirs)), cone_half_angle,
                       as_t(weights)).numpy()
