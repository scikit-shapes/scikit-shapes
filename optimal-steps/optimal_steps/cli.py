"""Command-line interface: `python register.py --help`, or `register-shapes` once installed."""

import argparse
import contextlib
import io
import logging
import time

from .api import register
from .config import RegistrationConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diffeomorphic registration (LDDMM) for meshes, point clouds and volumes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument(
        "--source", required=True, help="Source mesh / point cloud / volume path"
    )
    parser.add_argument(
        "--target", required=True, help="Target mesh / point cloud / volume path"
    )

    # Data
    parser.add_argument(
        "--source_label",
        type=int,
        default=None,
        help="Label to extract if source is a volume",
    )
    parser.add_argument(
        "--target_label",
        type=int,
        default=None,
        help="Label to extract if target is a volume",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=None,
        help="Max points for target after decimation",
    )
    parser.add_argument(
        "--output", type=str, default="result.ply", help="Output path for deformed mesh"
    )

    # Algorithm
    parser.add_argument("--sigma_init", type=float, default=30.0)
    parser.add_argument("--sigma_final", type=float, default=7.0)
    parser.add_argument(
        "--sigmas", type=float, nargs="+", default=None, help="Explicit sigma schedule"
    )
    parser.add_argument("--n_scales", type=int, default=4)
    parser.add_argument("--outer_steps", type=int, default=4)
    parser.add_argument("--lambda_reg", type=float, nargs="+", default=[0.5])
    parser.add_argument(
        "--metric_type",
        type=str,
        default="plane2plane",
        choices=["point2point", "point2plane", "plane2plane"],
    )
    parser.add_argument("--use_fpfh", action="store_true")
    parser.add_argument("--fpfh_weight", type=float, nargs="+", default=[0.0])
    parser.add_argument("--fpfh_radius", type=float, default=10.0)
    parser.add_argument("--normal_weight", type=float, default=0.1)
    parser.add_argument("--use_symmetric", action="store_true")
    parser.add_argument("--trust_symmetric", type=float, nargs="+", default=[0.0])
    parser.add_argument(
        "--incompressibility_weight", type=float, nargs="+", default=[0.0]
    )
    parser.add_argument("--solver_precision_mm", type=float, default=1e-4)
    parser.add_argument("--euler_precision_step_mm", type=float, default=1.0)
    parser.add_argument(
        "--no_rigid", action="store_true", help="Disable rigid RANSAC+ICP pre-alignment"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="auto (default): CUDA if available, CPU otherwise",
    )
    parser.add_argument("--quiet", action="store_true", help="Only print the output path and the timings")

    return parser


def _scalar_or_list(val):
    if isinstance(val, list) and len(val) == 1:
        return val[0]
    return val


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.quiet:
        logging.basicConfig(level=logging.WARNING)
    else:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    config = RegistrationConfig(
        sigma_init=args.sigma_init,
        sigma_final=args.sigma_final,
        sigmas=args.sigmas,
        n_scales=args.n_scales,
        outer_steps=args.outer_steps,
        lambda_reg=_scalar_or_list(args.lambda_reg),
        metric_type=args.metric_type,
        use_fpfh=args.use_fpfh,
        fpfh_weight=_scalar_or_list(args.fpfh_weight),
        fpfh_radius=args.fpfh_radius,
        normal_weight=args.normal_weight,
        use_symmetric_correspondences=args.use_symmetric,
        trust_symmetric=_scalar_or_list(args.trust_symmetric),
        incompressibility_weight=_scalar_or_list(args.incompressibility_weight),
        solver_precision_mm=args.solver_precision_mm,
        euler_precision_step_mm=args.euler_precision_step_mm,
        device=args.device,
    )

    t_total = time.perf_counter()
    # --quiet: hide the progress messages printed during the registration
    progress = contextlib.redirect_stdout(io.StringIO()) if args.quiet else contextlib.nullcontext()
    with progress:
        result = register(
            source=args.source,
            target=args.target,
            config=config,
            source_label=args.source_label,
            target_label=args.target_label,
            max_points=args.max_points,
            rigid_align=not args.no_rigid,
        )
    elapsed = time.perf_counter() - t_total

    out_mesh = result.source_mesh.copy()
    out_mesh.points = result.deformed_points
    out_mesh.save(args.output)
    print(f"Saved deformed mesh to {args.output}")

    print("\nTimings:")
    for k, v in result.timings.items():
        print(f"  {k}: {v:.3f}s")
    print(f"  TOTAL: {elapsed:.3f}s")


if __name__ == "__main__":
    main()
