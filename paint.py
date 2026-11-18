"""How even is the paint film of the PaintNet cuboid paths?

  python paint.py part 0 --standoff 120 --refine   one part, plotted
  python paint.py report sweep challenge           regenerate figures in outputs/
"""
import argparse

import matplotlib.pyplot as plt

from src import report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    part = commands.add_parser("part", help="simulate one part and plot its film")
    part.add_argument("index", type=int)
    part.add_argument("--standoff", type=float, default=200.0, help="mm")
    part.add_argument("--cone", type=float, default=25.0, help="half-angle in degrees")
    part.add_argument("--refine", action="store_true", help="also refine the path")
    run = commands.add_parser("report", help="regenerate figures and tables (all by default)")
    run.add_argument("sections", nargs="*", choices=list(report.SECTIONS),
                     default=list(report.SECTIONS))
    args = parser.parse_args()

    if args.command == "part":
        report.part(args.index, args.standoff, args.cone, args.refine)
        plt.show()
    else:
        for name in args.sections:
            report.SECTIONS[name]()
