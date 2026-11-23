import itertools

import numpy as np
import pytest
import torch

from src.data import load_sample
from src.metrics import paint_coverage, stripe_cv
from src.refine import refine
from src.report import flow_to_pass, sloppy_path
from src.spray import deposit, film, gun_poses, sample_surface, spray_directions


def box(size):
    """Closed, outward-wound triangle mesh of an axis-aligned box."""
    size = np.asarray(size, dtype=float)
    corners = np.array(list(itertools.product([0, 1], repeat=3)), dtype=float) * size
    faces = []
    for axis, side in itertools.product(range(3), (0, 1)):
        q = [i for i, c in enumerate(corners) if c[axis] == side * size[axis]]
        for tri in ([q[0], q[1], q[3]], [q[0], q[3], q[2]]):
            p = corners[tri]
            faces.append(tri if np.cross(p[1] - p[0], p[2] - p[0]) @ (p.mean(0) - size / 2) > 0
                         else tri[::-1])
    return corners, np.array(faces)


@pytest.fixture
def plate():
    """A flat box with its top rastered in passes 100 mm apart, spraying straight down."""
    vertices, faces = box([400.0, 300.0, 50.0])
    x = np.arange(0, 401, 2.0)
    xyz = np.vstack([np.column_stack([x, np.full_like(x, y), np.full_like(x, 50.0)])
                     for y in (50.0, 150.0, 250.0)])
    stroke = np.column_stack([xyz, np.tile([0.0, 90.0, 0.0], (len(xyz), 1))])
    return {"vertices": vertices, "faces": faces, "strokes": [stroke]}


def test_spray_directions_match_the_authors_convention():
    rotation = pytest.importorskip("scipy.spatial.transform").Rotation
    abc = np.random.default_rng(0).uniform(-360, 360, (1000, 3))
    expected = rotation.from_euler("yzx", abc[:, [1, 2, 0]], degrees=True).apply([1, 0, 0])
    np.testing.assert_allclose(spray_directions(abc), expected, atol=1e-12)


def test_surface_normals_point_outward_whatever_the_winding():
    vertices, faces = box([400.0, 300.0, 50.0])
    for f in (faces, faces[:, ::-1]):
        points, normals = sample_surface(vertices, f, 5000)
        assert ((points - [200, 150, 25]) * normals).sum(1).min() > 0


def test_deposit_cone_falloff_and_back_faces():
    g = np.arange(-150.0, 151, 5)
    points = np.column_stack([*map(np.ravel, np.meshgrid(g, g)), np.zeros(g.size**2)])
    up = np.tile([0.0, 0, 1], (len(points), 1))
    down = np.array([[0.0, 0, -1]])
    t = film(points, up, np.array([[0.0, 0, 200]]), down)
    r = np.linalg.norm(points[:, :2], axis=1)
    assert r[t.argmax()] == 0
    assert (t[r > 200 * np.tan(np.radians(25)) + 1] == 0).all()
    origin, near, far = np.zeros((1, 3)), np.array([[0.0, 0, 100]]), np.array([[0.0, 0, 200]])
    assert film(origin, up[:1], near, down)[0] == pytest.approx(4 * film(origin, up[:1], far, down)[0])
    assert film(origin, -up[:1], near, down)[0] == 0


def test_soft_edge_gradient_matches_finite_difference(plate):
    points, normals = map(torch.as_tensor, sample_surface(plate["vertices"], plate["faces"], 2000))
    guns, dirs = map(torch.as_tensor, gun_poses(plate["strokes"], step=20))

    def cv(lift):
        t = deposit(points, normals, guns + lift * dirs.new_tensor([0, 0, 1]), dirs, edge_width=1.0)
        return t.std(correction=0) / t.mean()

    lift = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    cv(lift).backward()
    numeric = (cv(torch.tensor(1e-3)) - cv(torch.tensor(-1e-3))) / 2e-3
    assert lift.grad.item() == pytest.approx(numeric.item(), rel=1e-4)


def test_paint_coverage_ignores_excess_paint():
    reference = np.linspace(1, 2, 1001)
    assert paint_coverage(reference, reference) == paint_coverage(5 * reference, reference) == 1
    assert paint_coverage(0.5 * reference, reference) < 0.5


def test_stripe_cv_sees_stripes_along_z_only():
    points, normals = sample_surface(*box([400.0, 300.0, 600.0]), 20000)
    along_z = 1 + 0.5 * np.sin(2 * np.pi * points[:, 2] / 100)
    along_x = 1 + 0.5 * np.sin(2 * np.pi * points[:, 0] / 100)
    assert stripe_cv(points, normals, along_z) > 0.25
    assert stripe_cv(points, normals, along_x) < stripe_cv(points, normals, along_z) / 3


def test_sloppy_path_shifts_the_aim_sideways_by_the_error(plate):
    guns, dirs = sloppy_path(plate, 200.0, error=10.0)
    offset = guns - gun_poses(plate["strokes"], 200.0)[0]
    np.testing.assert_allclose((offset * dirs).sum(1), 0, atol=1e-9)
    assert (np.abs(offset[:, 0]) < 1e-9).mean() > 0.95  # passes run along X
    assert np.sqrt((offset**2).sum(1).mean()) == pytest.approx(10)


def test_flow_to_pass_is_the_smallest_flow_that_passes():
    rng = np.random.default_rng(0)
    reference = rng.uniform(0.8, 1.2, 5000)
    thickness = reference * rng.lognormal(0, 0.6, 5000)
    flow = flow_to_pass(thickness, reference)
    assert paint_coverage(flow * thickness, reference) >= 0.99
    assert paint_coverage(0.95 * flow * thickness, reference) < 0.99


def test_refine_evens_the_film_within_bounds(plate):
    r = refine(plate, steps=20, n_points=1500, n_eval=3000, device="cpu", log_every=0)
    top = r["normals"][:, 2] > 0.95
    spread = lambda t: t[top].std() / t[top].mean()
    assert spread(r["after"]) < spread(r["before"])
    assert np.abs(r["trajectory"][:, 6] - 200).max() <= 50
    assert r["trajectory"][:, 7].min() > 0


def test_load_sample(tmp_path):
    for name in ("10_cube", "2_cube"):
        (tmp_path / name).mkdir()
        (tmp_path / name / f"{name}.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1/1/1 2/2/2 3/3/3\n")
        (tmp_path / name / "trajectory.txt").write_text("X;Y;Z;A;B;C;strokeId\n0;0;0;0;0;0;1\n5;5;5;0;90;0;0\n")
    sample = load_sample(1, tmp_path)
    assert sample["name"] == "10_cube"
    np.testing.assert_array_equal(sample["faces"], [[0, 1, 2]])
    np.testing.assert_array_equal(sample["strokes"][0], [[5, 5, 5, 0, 90, 0]])
