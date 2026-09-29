"""Vertebra registration on the VerSe dataset and atlases (paper, Section 3).

For every vertebra label, the template `data/templates_vertebrae/template_<label>.ply`
is registered onto all the VerSe segmentations that contain this label, and the
metrics of Table 1 (Chamfer, HD95, log-Jacobian variance) are reported for every
registration. The registered shapes then give an atlas per label. Since
every registered mesh shares the template's triangulation, the atlas is then a
plain per-vertex mean, and the modes of variation a PCA of the vertex positions.
The rigid pre-alignment is undone before averaging, so that all the shapes are
expressed in the template's frame.

    # C7 only, on the VerSe test split
    python register_vertebrae.py --dataset_root data/verse --parts dataset-03test --labels 7

    # all 24 labels (the paper's atlas: 978 test vertebrae)
    python register_vertebrae.py --dataset_root data/verse --parts dataset-03test

    # only recompute the mean and the modes from registrations done earlier
    python register_vertebrae.py --labels 7 --skip_registration

Outputs, in `<output_dir>/L<label>/`:

    manifest.csv, results.csv, <case>/...   registrations (see register_batch.py)
    atlas_mean.ply                          mean shape
    atlas_mode<k>_{minus,plus}<n>sd.ply     mean -/+ n standard deviations along mode k
    atlas_pca.npz                           mean, modes, standard deviations, explained variance

Registrations already present in results.csv are not redone, so an interrupted
run can simply be relaunched.
"""

import argparse
import csv
import json
import zipfile
from pathlib import Path

import numpy as np
import pyvista as pv

from register_batch import (
    add_config_arguments,
    config_from_args,
    parse_max_points,
    run_manifest,
)

# The 24 templates are archived on Zenodo (doi:10.5281/zenodo.22940731); only those of the
# examples are in the repository. The missing ones are downloaded on the first run (and
# cached by pooch).
TEMPLATES_URL = "https://zenodo.org/records/22940731/files/templates_vertebrae.zip?download=1"
TEMPLATES_SHA256 = "2bd59335600e3554136481ae8c1ec7275d425fc8c4611358056921b8ef1bb73c"

VERTEBRA_NAMES = (
    [f"C{i}" for i in range(1, 8)] + [f"T{i}" for i in range(1, 13)] + [f"L{i}" for i in range(1, 6)]
)


def ensure_templates(templates_dir: str, labels: list):
    """Download the templates missing from `templates_dir` (first run only)."""
    templates_dir = Path(templates_dir)
    missing = [label for label in labels if not (templates_dir / f"template_{label}.ply").exists()]
    if not missing:
        return
    if TEMPLATES_URL is None:
        raise SystemExit(
            f"Templates of labels {missing} not found in {templates_dir}/, and no download "
            "source is configured (TEMPLATES_URL in register_vertebrae.py)."
        )
    import pooch

    print(f"Downloading the vertebra templates from {TEMPLATES_URL} ...")
    archive = pooch.retrieve(TEMPLATES_URL, known_hash=f"sha256:{TEMPLATES_SHA256}", progressbar=False)
    templates_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            filename = Path(name).name
            target = templates_dir / filename
            if filename.startswith("template_") and filename.endswith(".ply") and not target.exists():
                target.write_bytes(z.read(name))
    print(f"Templates saved in {templates_dir}/")


def find_verse_scans(dataset_root: str, parts: list) -> list:
    """(mask path, scan name, labels present) for every VerSe scan in `parts`.

    Labels are read from the centroid JSON next to each mask (VerSe layout:
    <part>/derivatives/sub-*/<scan>_seg-vert_msk.nii.gz + <scan>_seg-subreg_ctd.json).
    """
    scans = []
    for part in parts:
        part_dir = Path(dataset_root) / part
        search_dir = part_dir / "derivatives" if (part_dir / "derivatives").exists() else part_dir
        for mask in sorted(search_dir.glob("sub-*/*_seg-vert_msk.nii.gz")):
            scan = mask.name[: -len("_seg-vert_msk.nii.gz")]
            ctd = mask.with_name(f"{scan}_seg-subreg_ctd.json")
            if not ctd.exists():
                print(f"[warning] no centroid file for {mask}, skipped")
                continue
            with open(ctd) as f:
                labels = {int(item["label"]) for item in json.load(f) if "label" in item}
            scans.append((mask, scan, labels))
    return scans


def write_manifest(path: Path, rows: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "target", "source_label", "target_label", "id"])
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ("" if r[k] is None else r[k]) for k in writer.fieldnames})


def registered_meshes(label_dir: Path) -> list:
    """Meshes in the template frame for the successful rows of results.csv."""
    results = label_dir / "results.csv"
    if not results.exists():
        return []
    with open(results, newline="") as f:
        ids = [r["id"] for r in csv.DictReader(f) if r["status"] == "ok"]
    paths = [label_dir / i / "deformed_in_source_frame.ply" for i in ids]
    return [p for p in paths if p.exists()]


def compute_atlas(mesh_paths: list, out_dir: Path, n_modes: int = 3, n_sd: float = 2.0):
    """Mean shape and PCA modes of meshes sharing the same triangulation."""
    reference = pv.read(str(mesh_paths[0]))
    shapes = [pv.read(str(p)).points.reshape(-1) for p in mesh_paths]
    if any(len(s) != len(shapes[0]) for s in shapes):
        raise ValueError("All meshes must share the template's vertices.")
    shapes = np.stack(shapes).astype(np.float64)

    mean = shapes.mean(axis=0)
    # PCA through the SVD of the centred data matrix (n_shapes x 3N).
    _, singular_values, components = np.linalg.svd(shapes - mean, full_matrices=False)
    variances = singular_values**2 / max(1, len(shapes) - 1)
    explained = variances / variances.sum() if variances.sum() > 0 else variances
    n_modes = min(n_modes, len(shapes) - 1)

    def save(points, name):
        mesh = reference.copy()
        mesh.points = points.reshape(-1, 3).astype(np.float32)
        mesh.save(str(out_dir / name), binary=True)

    save(mean, "atlas_mean.ply")
    for k in range(n_modes):
        sd = np.sqrt(variances[k])
        tag = f"{n_sd:g}sd"
        save(mean - n_sd * sd * components[k], f"atlas_mode{k + 1}_minus{tag}.ply")
        save(mean + n_sd * sd * components[k], f"atlas_mode{k + 1}_plus{tag}.ply")

    np.savez(
        out_dir / "atlas_pca.npz",
        mean=mean.reshape(-1, 3),
        modes=components[: max(n_modes, 0)].reshape(-1, len(mean) // 3, 3),
        std=np.sqrt(variances[: max(n_modes, 0)]),
        explained_variance_ratio=explained,
        faces=np.asarray(reference.faces),
    )
    return explained[:n_modes]


def main():
    parser = argparse.ArgumentParser(
        description="VerSe vertebra registration (metrics of Table 1) and atlases (mean shape + PCA modes)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_root", default="data/verse")
    parser.add_argument("--parts", nargs="+", default=["dataset-03test"], help="VerSe splits to use")
    parser.add_argument("--templates_dir", default="data/templates_vertebrae")
    parser.add_argument("--labels", type=int, nargs="+", default=list(range(1, 25)), help="1-7: C1-C7, 8-19: T1-T12, 20-24: L1-L5")
    parser.add_argument("--output_dir", default="results/vertebrae")
    parser.add_argument("--n_modes", type=int, default=3, help="PCA modes to export")
    parser.add_argument("--n_sd", type=float, default=2.0, help="Standard deviations for the exported modes")
    parser.add_argument("--min_shapes", type=int, default=2, help="Skip labels with fewer registered shapes")
    parser.add_argument("--skip_registration", action="store_true", help="Only compute the atlases from existing registrations")
    parser.add_argument("--no_metrics", action="store_true", help="Skip Chamfer / HD95 / log-Jacobian")
    add_config_arguments(parser)
    args = parser.parse_args()

    config = config_from_args(args)
    if not args.skip_registration:
        ensure_templates(args.templates_dir, args.labels)
    scans = [] if args.skip_registration else find_verse_scans(args.dataset_root, args.parts)
    if not args.skip_registration:
        print(f"{len(scans)} VerSe scans found in {args.dataset_root} ({', '.join(args.parts)})")

    summary = []
    for label in args.labels:
        name = VERTEBRA_NAMES[label - 1] if 1 <= label <= len(VERTEBRA_NAMES) else str(label)
        label_dir = Path(args.output_dir) / f"L{label}"
        print(f"\n=== Label {label} ({name}) ===")

        if not args.skip_registration:
            template = Path(args.templates_dir) / f"template_{label}.ply"
            if not template.exists():
                print(f"[warning] {template} not found, label skipped")
                continue
            rows = [
                dict(source=str(template.resolve()), target=str(mask.resolve()),
                     source_label=None, target_label=label, id=f"{scan}_L{label}")
                for mask, scan, labels in scans
                if label in labels
            ]
            if not rows:
                print("no scan contains this label, skipped")
                continue
            write_manifest(label_dir / "manifest.csv", rows)
            run_manifest(
                rows,
                config,
                label_dir,
                max_points=parse_max_points(args.max_points),
                rigid_align=not args.no_rigid,
                compute_metrics=not args.no_metrics,
                skip_existing=True,
            )

        meshes = registered_meshes(label_dir)
        if len(meshes) < args.min_shapes:
            print(f"only {len(meshes)} registered shapes, no atlas for this label")
            continue
        explained = compute_atlas(meshes, label_dir, args.n_modes, args.n_sd)
        modes = ", ".join(f"{100 * e:.1f}%" for e in explained)
        print(f"atlas from {len(meshes)} shapes saved to {label_dir}/ (variance of the first modes: {modes})")
        summary.append((label, name, len(meshes)))

    if summary:
        total = sum(n for _, _, n in summary)
        print(f"\n{len(summary)} atlases computed from {total} registered vertebrae in {args.output_dir}/")


if __name__ == "__main__":
    main()
