"""Register many source/target pairs listed in a CSV manifest.

Manifest columns (one row per registration):

    source        path to the source mesh / segmentation        (required)
    target        path to the target mesh / segmentation        (required)
    source_label  label to extract if the source is a volume    (optional)
    target_label  label to extract if the target is a volume    (optional)
    id            name of the output folder for this row        (optional)

Relative paths are resolved from the manifest's folder. Example:

    python register_batch.py --manifest examples/manifest_example.csv --output_dir results/batch

For every row, `<output_dir>/<id>/` receives:

    source.ply                   source surface actually used (after extraction), in its own frame
    source_prealigned.ply        source after the rigid + anisotropic pre-alignment (target frame)
    deformed.ply                 source deformed onto the target (target frame)
    deformed_in_source_frame.ply same mesh with the rigid pre-alignment undone,
                                 i.e. in the frame of the source (used for atlases)
    target.ply                   target surface actually used (after extraction/decimation)

and `<output_dir>/results.csv` summarises timings and the paper's metrics
(Chamfer distance, HD95, log-Jacobian variance) for every row. A failing row is
reported in results.csv and does not stop the batch.
"""

import argparse
import csv
import logging
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np

from optimal_steps import RegistrationConfig, load_input, register, resolve_device
from optimal_steps.metrics import evaluate_registration


# Configuration used for the VerSe experiments of the paper.
VERSE_CONFIG = dict(
    sigma_init=10.0,
    sigma_final=4.0,
    n_scales=4,
    outer_steps=4,
    lambda_reg=0.5,
    use_fpfh=True,
    fpfh_weight=[0.1, 0.3, 0.0, 0.0],
    fpfh_radius=10.0,
    normal_weight=0.05,
    use_symmetric_correspondences=True,
    trust_symmetric=0.7,
    solver_precision_mm=0.5,
)

RESULT_COLUMNS = [
    "id",
    "status",
    "source",
    "target",
    "source_label",
    "target_label",
    "n_source_points",
    "n_target_points",
    "time_rigid_s",
    "time_non_rigid_s",
    "time_total_s",
    "chamfer_mm",
    "hd95_mm",
    "log_jacobian_var",
    "error",
]


def _optional_int(value) -> Optional[int]:
    if value is None or str(value).strip() in ("", "none", "None", "nan"):
        return None
    return int(float(value))


def _default_id(target: str, target_label: Optional[int]) -> str:
    name = Path(target).name
    for suffix in (".nii.gz", ".nii", ".ply", ".vtk", ".stl", ".obj", ".vtp"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name if target_label is None else f"{name}_L{target_label}"


def read_manifest(path: str) -> list:
    """Read the manifest into a list of dicts with resolved paths and unique ids."""
    manifest_dir = Path(path).resolve().parent

    def resolve(p: str) -> str:
        p = Path(p.strip())
        return str(p if p.is_absolute() else (manifest_dir / p).resolve())

    rows, seen = [], {}
    with open(path, newline="") as f:
        for i, raw in enumerate(csv.DictReader(f)):
            raw = {k.strip(): (v or "").strip() for k, v in raw.items() if k}
            if not raw.get("source") or not raw.get("target"):
                raise ValueError(f"{path}, row {i + 2}: 'source' and 'target' are required.")
            row = {
                "source": resolve(raw["source"]),
                "target": resolve(raw["target"]),
                "source_label": _optional_int(raw.get("source_label")),
                "target_label": _optional_int(raw.get("target_label")),
            }
            row_id = raw.get("id") or _default_id(row["target"], row["target_label"])
            if row_id in seen:  # keep output folders distinct
                seen[row_id] += 1
                row_id = f"{row_id}_{seen[row_id]}"
            else:
                seen[row_id] = 0
            row["id"] = row_id
            rows.append(row)
    return rows


def undo_rigid(points: np.ndarray, rigid_transform: Optional[np.ndarray]) -> np.ndarray:
    """Map points from the target frame back to the source frame."""
    if rigid_transform is None:
        return points.astype(np.float32)
    inv = np.linalg.inv(rigid_transform)
    return (points @ inv[:3, :3].T + inv[:3, 3]).astype(np.float32)


class _SourceCache:
    """Load each source once: in an atlas, all rows share the same template."""

    def __init__(self):
        self._cache = {}

    def get(self, path: str, label: Optional[int]):
        key = (path, label)
        if key not in self._cache:
            self._cache[key] = load_input(path, label=label, max_points=None)
        mesh, has_connectivity = self._cache[key]
        return mesh.copy(), has_connectivity


def register_row(
    row: dict,
    config: RegistrationConfig,
    output_dir: Path,
    sources: _SourceCache,
    max_points: Optional[int],
    rigid_align: bool = True,
    compute_metrics: bool = True,
) -> dict:
    out = {k: row.get(k) for k in ("id", "source", "target", "source_label", "target_label")}
    case_dir = output_dir / row["id"]
    case_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    source_mesh, source_is_mesh = sources.get(row["source"], row["source_label"])
    # register() treats a PolyData input as a triangle mesh, so point clouds
    # are passed by path to keep its point-cloud handling.
    source_input = source_mesh.copy() if source_is_mesh else row["source"]

    result = register(
        source=source_input,
        target=row["target"],
        config=config,
        source_label=row["source_label"],
        target_label=row["target_label"],
        max_points=max_points,
        rigid_align=rigid_align,
    )
    total = time.perf_counter() - t0

    deformed = result.deformed_mesh
    source_mesh.save(str(case_dir / "source.ply"), binary=True)
    result.source_mesh.save(str(case_dir / "source_prealigned.ply"), binary=True)
    deformed.save(str(case_dir / "deformed.ply"), binary=True)
    result.target_mesh.save(str(case_dir / "target.ply"), binary=True)
    in_source_frame = deformed.copy()
    in_source_frame.points = undo_rigid(result.deformed_points, result.rigid_transform)
    in_source_frame.save(str(case_dir / "deformed_in_source_frame.ply"), binary=True)

    out.update(
        status="ok",
        n_source_points=result.source_mesh.n_points,
        n_target_points=result.target_mesh.n_points,
        time_rigid_s=result.timings.get("rigid_alignment"),
        time_non_rigid_s=result.timings.get("non_rigid"),
        time_total_s=total,
    )
    if compute_metrics:
        # Log-Jacobian against the source *before* pre-alignment, as in the paper.
        out.update(evaluate_registration(deformed, result.target_mesh, source_mesh))
    return out


def run_manifest(
    rows: list,
    config: RegistrationConfig,
    output_dir: str,
    max_points: Optional[int] = 10000,
    rigid_align: bool = True,
    compute_metrics: bool = True,
    skip_existing: bool = False,
    results_name: str = "results.csv",
) -> list:
    """Register every row, writing results.csv as the batch progresses."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / results_name
    sources = _SourceCache()
    resolve_device(config.device)  # announce CPU / GPU once, before the first row

    previous = {}
    if skip_existing and results_path.exists():
        with open(results_path, newline="") as f:
            previous = {r["id"]: r for r in csv.DictReader(f) if r["status"] == "ok"}

    results = []
    for i, row in enumerate(rows):
        prefix = f"[{i + 1}/{len(rows)}] {row['id']}"
        if row["id"] in previous and (output_dir / row["id"] / "deformed.ply").exists():
            print(f"{prefix}: already done, skipped")
            results.append(previous[row["id"]])
            continue
        print(f"{prefix}: registering...", flush=True)
        try:
            res = register_row(
                row, config, output_dir, sources, max_points, rigid_align, compute_metrics
            )
            metrics = ", ".join(
                f"{k}={res[k]:.3f}" for k in ("chamfer_mm", "hd95_mm", "log_jacobian_var") if k in res
            )
            print(f"{prefix}: done in {res['time_total_s']:.2f}s  {metrics}")
        except Exception as exc:  # one bad case must not stop a long batch
            logging.getLogger(__name__).debug(traceback.format_exc())
            print(f"{prefix}: FAILED ({type(exc).__name__}: {exc})")
            res = {k: row.get(k) for k in ("id", "source", "target", "source_label", "target_label")}
            res.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        results.append(res)
        _write_results(results_path, results)

    _print_summary(results)
    print(f"Results written to {results_path}")
    return results


def _write_results(path: Path, results: list):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in RESULT_COLUMNS})


def _print_summary(results: list):
    ok = [r for r in results if r.get("status") == "ok"]
    print(f"\n{len(ok)}/{len(results)} registrations succeeded.")
    for key in ("time_non_rigid_s", "chamfer_mm", "hd95_mm", "log_jacobian_var"):
        vals = [float(r[key]) for r in ok if r.get(key) not in (None, "")]
        if vals:
            print(f"  {key:18s} {np.mean(vals):.3f} ± {np.std(vals):.3f}")


def config_from_args(args) -> RegistrationConfig:
    def scalar_or_list(v):
        return v[0] if isinstance(v, list) and len(v) == 1 else v

    return RegistrationConfig(
        sigma_init=args.sigma_init,
        sigma_final=args.sigma_final,
        n_scales=args.n_scales,
        outer_steps=args.outer_steps,
        lambda_reg=args.lambda_reg,
        use_fpfh=not args.no_fpfh,
        fpfh_weight=scalar_or_list(args.fpfh_weight),
        fpfh_radius=args.fpfh_radius,
        normal_weight=args.normal_weight,
        use_symmetric_correspondences=args.trust_symmetric > 0,
        trust_symmetric=args.trust_symmetric,
        solver_precision_mm=args.solver_precision_mm,
        device=args.device,
    )


def add_config_arguments(parser: argparse.ArgumentParser):
    """Registration hyperparameters, defaulting to the paper's VerSe setting."""
    c = VERSE_CONFIG
    g = parser.add_argument_group("registration parameters (defaults: VerSe setting of the paper)")
    g.add_argument("--sigma_init", type=float, default=c["sigma_init"])
    g.add_argument("--sigma_final", type=float, default=c["sigma_final"])
    g.add_argument("--n_scales", type=int, default=c["n_scales"])
    g.add_argument("--outer_steps", type=int, default=c["outer_steps"])
    g.add_argument("--lambda_reg", type=float, default=c["lambda_reg"])
    g.add_argument("--no_fpfh", action="store_true", help="Disable FPFH descriptors")
    g.add_argument("--fpfh_weight", type=float, nargs="+", default=c["fpfh_weight"])
    g.add_argument("--fpfh_radius", type=float, default=c["fpfh_radius"])
    g.add_argument("--normal_weight", type=float, default=c["normal_weight"])
    g.add_argument(
        "--trust_symmetric",
        type=float,
        default=c["trust_symmetric"],
        help="Weight of target-to-source matches (0 disables symmetric matching)",
    )
    g.add_argument("--solver_precision_mm", type=float, default=c["solver_precision_mm"])
    g.add_argument("--max_points", type=str, default="10000", help="Target decimation ('none' to keep all)")
    g.add_argument("--no_rigid", action="store_true", help="Disable RANSAC+ICP pre-alignment")
    g.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])


def parse_max_points(value: str) -> Optional[int]:
    return None if str(value).lower() == "none" else int(value)


def main():
    parser = argparse.ArgumentParser(
        description="Batch registration from a CSV manifest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", required=True, help="CSV with source,target[,source_label,target_label,id]")
    parser.add_argument("--output_dir", default="results/batch")
    parser.add_argument("--no_metrics", action="store_true", help="Skip Chamfer / HD95 / log-Jacobian")
    parser.add_argument("--skip_existing", action="store_true", help="Resume: skip rows already in results.csv")
    add_config_arguments(parser)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    rows = read_manifest(args.manifest)
    print(f"{len(rows)} registrations listed in {args.manifest}")
    run_manifest(
        rows,
        config_from_args(args),
        args.output_dir,
        max_points=parse_max_points(args.max_points),
        rigid_align=not args.no_rigid,
        compute_metrics=not args.no_metrics,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
