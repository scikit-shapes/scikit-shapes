"""Register one source onto one target: the simplest use of register().

    # C7 template onto the C7 vertebra of the VerSe scan included in the repository
    python register_pair.py

    # your own data (meshes, point clouds, or segmentations with a label)
    python register_pair.py --source my_source.ply --target my_segmentation.nii.gz --target_label 3

Writes, in --output_dir: source.ply (source as loaded), source_prealigned.ply
(after the rigid + anisotropic pre-alignment), deformed.ply (source deformed onto
the target) and target.ply (target surface actually used), then prints the
timings and the metrics of the paper. --show opens a 3D view of the result.

The registration parameters default to the VerSe setting of the paper
(see `python register_pair.py --help`).
"""

import argparse
import time
from pathlib import Path

from optimal_steps import load_input, register
from register_batch import add_config_arguments, config_from_args, parse_max_points
from optimal_steps.metrics import evaluate_registration

EXAMPLE_SOURCE = "data/templates_vertebrae/template_7.ply"
EXAMPLE_TARGET = "data/verse/dataset-01training/derivatives/sub-gl090/sub-gl090_dir-ax_seg-vert_msk.nii.gz"


def show_result(source, deformed, target):
    import pyvista as pv

    pl = pv.Plotter(shape=(1, 2), window_size=(1600, 800))
    for col, (mesh, title) in enumerate(((source, "Before (pre-aligned)"), (deformed, "After registration"))):
        pl.subplot(0, col)
        pl.add_mesh(mesh, color=(0.85, 0.15, 0.15), smooth_shading=True, label="source")
        pl.add_mesh(target, color=(0.33, 0.61, 0.92), opacity=0.35, label="target")
        pl.add_title(title, font_size=14)
        pl.add_legend()
        pl.enable_parallel_projection()
    pl.link_views()
    pl.show()


def main():
    parser = argparse.ArgumentParser(
        description="Register one source onto one target.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", default=EXAMPLE_SOURCE, help="Source mesh / point cloud / segmentation")
    parser.add_argument("--target", default=EXAMPLE_TARGET, help="Target mesh / point cloud / segmentation")
    parser.add_argument("--source_label", type=int, default=None, help="Label to extract if the source is a segmentation")
    parser.add_argument("--target_label", type=int, default=None,
                        help="Label to extract if the target is a segmentation (7 for the default example)")
    parser.add_argument("--output_dir", default="results/pair")
    parser.add_argument("--show", action="store_true", help="Open a 3D view of the result")
    add_config_arguments(parser)
    args = parser.parse_args()

    target_label = args.target_label
    if target_label is None and args.target == EXAMPLE_TARGET:
        target_label = 7  # C7

    config = config_from_args(args)
    source_mesh, _ = load_input(args.source, label=args.source_label)

    t0 = time.perf_counter()
    result = register(
        source=args.source,
        target=args.target,
        config=config,
        source_label=args.source_label,
        target_label=target_label,
        max_points=parse_max_points(args.max_points),
        rigid_align=not args.no_rigid,
    )
    elapsed = time.perf_counter() - t0

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    source_mesh.save(str(out / "source.ply"))
    result.source_mesh.save(str(out / "source_prealigned.ply"))
    result.deformed_mesh.save(str(out / "deformed.ply"))
    result.target_mesh.save(str(out / "target.ply"))

    print(f"\nRegistration done in {elapsed:.2f} s")
    for step, t in result.timings.items():
        print(f"  {step:16s} {t:.2f} s")
    for name, value in evaluate_registration(result.deformed_mesh, result.target_mesh, source_mesh).items():
        print(f"  {name:16s} {value:.3f}")
    print(f"Meshes saved in {out}/")

    if args.show:
        show_result(result.source_mesh, result.deformed_mesh, result.target_mesh)


if __name__ == "__main__":
    main()
