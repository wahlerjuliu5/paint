# Paint Evenness

PaintNet and MaskPlanner (Tiboni et al.) report up to 99% paint coverage on the PaintNet
cuboids. Their coverage counts a point once its film passes a bar taken from the expert
path, so it cannot see uneven films or excess paint. This repo simulates the same paths
and measures evenness, stripes from the pass spacing and the share of surface within
±20% of target. It also refines paths by gradient descent through the simulator.

```bash
conda activate paint
pip install -e ".[test]"
python paint.py part 0 --standoff 120 --refine   # one part, plotted
python paint.py report                           # all figures and tables into outputs/
pytest
```

The cuboids come from the extended PaintNet dataset
([Zenodo 14967945](https://zenodo.org/records/14967945)) and go in `cuboids-v2/`. The
full report takes about an hour and a half, mostly refinement; name sections
(`stats`, `sweep`, `challenge`, `refinement`) to run only those.

## Assumptions

- **Orientation.** A, B, C in `trajectory.txt` are degrees: the spray axis is +X rotated by scipy's `from_euler("yzx", [B, C, A])`, as in the authors' loader.
- **Gun.** Recorded positions are aim points on the surface, and the gun sits `--standoff` mm behind them. The spray is a Gaussian cone, 25° half-angle by default. The papers state a 120 mm deposit distance but no cone angle, and the stripes depend on that choice.
- **Occlusion.** Back-facing points get no paint, which is exact for cuboids only.
