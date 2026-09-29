# Optimal Steps for Fast Diffeomorphic Shape Registration

Code accompanying the MICCAI paper:

> **Optimal Steps for Fast Diffeomorphic Shape Registration**  
> Hadrien Bigo-Balland, Jean Feydy & Tom Boeken.

---

## Installation

### With pip (into any existing environment)

```bash
pip install -e .
```

This installs the dependencies listed in `pyproject.toml` / `requirements.txt` and
puts the `optimal_steps` package on the import path, so `from optimal_steps import register, RegistrationConfig`
works from any directory (and a `register-shapes` CLI is installed).

If you already have a specific PyTorch/CUDA build, install it first
([pytorch.org](https://pytorch.org/get-started/locally/)); the pins here are loose
(`torch>=2.1`, `pykeops>=2.2,<3`, `open3d>=0.17`, `pyvista>=0.43`) and an existing
compatible install is kept.

### With conda (exact versions used for the paper)

```bash
conda env create -f environment.yml
conda activate diffeo_registration_env
pip install -e .
```

`environment.yml` pins the versions the experiments were run with: PyTorch 2.10,
[KeOps](https://www.kernel-operations.io/keops/) 2.3 (GPU kernel operations, compiled
on first run), Open3D 0.19 (FPFH features), PyVista 0.46 (mesh I/O and visualisation).

> **macOS note:** on import, `optimal_steps` points KeOps to a small wrapper around Apple clang
> (it drops GCC-only flags and adds the macOS SDK path, which otherwise fails with
> `'cmath' file not found`). Import `optimal_steps` before `pykeops` in your own scripts.
> Nothing changes on Linux, or if `CXX` is already set.

### CPU or GPU

The code runs on both. `RegistrationConfig.device` defaults to `"auto"`: CUDA when a GPU is
visible, CPU otherwise. The choice is printed once at the start of a run, e.g.

```
[info] no CUDA found, running on CPU - registrations will be 10 to 100 times slower than on a GPU
```

so a CPU run is never mistaken for a crash. Orders of magnitude:

| Registration | NVIDIA RTX 3090 | CPU (Apple M-series laptop) |
| -------------- | ----------------- | ----------------------------- |
| vertebra, ~10k points (4 scales x 4 steps) | ~1 s | ~25 s |
| lung vessel tree, 25k–60k points (paper setting) | ~25 s (14–70 s) | hours |
| lung vessel tree, 8k points, lighter setting of the notebook | a few seconds (estimate) | ~2.5 min |

Add to that a one-off KeOps compilation of ~1–2 min the first time the kernels are
built on a given machine. KeOps runs on CPU and NVIDIA GPUs (CUDA) only, with no Apple
Metal/MPS backend, so Apple Silicon machines run on CPU. Force a device with `RegistrationConfig(device="cpu")` or `--device cpu`.

---

## Files

| File | Purpose |
| ------ | --------- |
| `notebooks/registration_examples.ipynb` | Run the paper experiments on one vertebra (VerSe) and one lung vessel tree (Lung250M-4B), in Jupyter or Colab - **start here** |
| `register_pair.py` | Register one source/target pair - **try it on your own data** |
| `register_batch.py` | Register many pairs listed in a CSV manifest, with a metrics summary |
| `register_vertebrae.py` | Register all the VerSe vertebrae, compute the metrics of Table 1 and the atlases (mean shapes and modes of variation) |
| `register_lungs.py` | Register the Lung250M-4B vessel trees and compute the landmark error |
| `optimal_steps/` | The method, as a package (see below) |
| `register.py` | Minimal command-line interface to `register()` |
| `examples/manifest_example.csv` | Example manifest for `register_batch.py` |
| `data/templates_vertebrae/` | Templates C6, C7 and T1, for the examples. The 24 templates (1-7: C1-C7, 8-19: T1-T12, 20-24: L1-L5) are on Zenodo and downloaded automatically, see [Vertebra templates](#vertebra-templates). Derived from BodyParts3D (CC BY 4.0, see [License](#license)) |
| `data/verse/` | One VerSe scan (`sub-gl090`), for the examples; the downloaded VerSe splits go here too |
| `data/lungs/` | One Lung250M-4B case (056), for the notebook |
| `environment.yml` | Conda environment specification (exact paper versions) |
| `pyproject.toml` / `requirements.txt` | pip installation (`pip install -e .`, `pip install -e ".[notebook]"` for the notebook) |


The `optimal_steps` package follows the steps of Section 2 of the paper:

| Module | Content |
|--------|---------|
| `config.py` | `RegistrationConfig`, device selection (CPU / GPU) |
| `geometry.py` | Mesh utilities: vertex areas, edge lengths |
| `io.py` | `load_input()`: meshes, point clouds, surface extraction from segmentations |
| `alignment.py` | `align_rigid()`: rigid + anisotropic pre-alignment (RANSAC on FPFH, then ICP) |
| `matching.py` | FPFH descriptors, `effective_targets()`: target position of every source point |
| `losses.py` | Point-to-point, point-to-plane and plane-to-plane losses |
| `solver.py` | Preconditioned conjugate gradient |
| `model.py` | `DiffeomorphicRegistration`: the multiscale loop (matching, regularisation, deformation) |
| `api.py` | `register()`, `RegistrationResult` |
| `metrics.py` | Chamfer distance, HD95 and log-Jacobian variance (Table 1) |
| `cli.py` | Command-line interface |

---

## Quick start

The simplest way to use the code on your own data:

```python
from optimal_steps import register, RegistrationConfig

config = RegistrationConfig(
    sigma_init=10.0,   # initial kernel width (mm) - ~1/10 to 1/5 of the shape diameter
    sigma_final=4.0,   # final kernel width (mm)  - ~2 to 5× the average point spacing
    n_scales=4,        # number of coarse-to-fine levels
    outer_steps=4,     # Gauss-Newton steps per scale
    lambda_reg=0.5,    # regularisation weight λ, higher = stiffer (see "Key hyperparameters")
    use_fpfh=True,     # use FPFH geometric descriptors for matching
    fpfh_weight=[0.1, 0.3, 0.0, 0.0],  # weight of FPFH alignment term per scale (coarse → fine, relative to point distances)
    fpfh_radius=10.0,  # neighbourhood radius (mm) for FPFH computation
    normal_weight=0.05, # weight of normal alignment term (relative to point distances)
    use_symmetric_correspondences=True,  # use bidirectional correspondences (recommended)
    trust_symmetric=0.7,  # weight κ of the backward matches (0 = forward only), see "Key hyperparameters"
)

result = register(
    source="path/to/source.ply",   # mesh, point cloud or segmentation (see "Accepted input formats"),
    target="path/to/target.ply",   # or a pyvista.PolyData
    config=config,
    source_label=7,   # label index to extract (only for .nii.gz inputs)
    target_label=7,
    max_points=10000, # downsample target to this number of points (None for no downsampling)
    rigid_align=True, # pre-align with RANSAC + ICP
    # optional, see "Landmark-constrained registration" below:
    # source_landmark_indices=[i0, i1, i2], target_landmark_indices=[j0, j1, j2],
    # optional, your own per-point descriptors, see "Custom features" below:
    # source_features=..., target_features=...,
)

# Deformed source mesh
result.deformed_mesh   # pyvista.PolyData with updated vertex positions
result.deformed_points # numpy array (N, 3)
result.source_mesh     # source after the rigid + anisotropic pre-alignment
result.target_mesh     # target mesh
```

`register_pair.py` does exactly this from the command line, and saves the meshes and prints
the metrics:

```bash
python register_pair.py                      # C7 example included in the repository
python register_pair.py --source my_source.ply --target my_seg.nii.gz --target_label 3 --show
```

### Accepted input formats

| Format | Notes |
| -------- | ------- |
| `.ply` | Triangle mesh - used directly |
| `.nii.gz` | Binary or label segmentation, use `source_label` / `target_label` to select the label |
| `.vtk`, `.vtp`, `.stl`, `.obj` | Any mesh format supported by PyVista |
| `.xyz`, `.pts`, `.csv`, or a mesh file without faces | Point cloud: FPFH, normals and plane metrics are then disabled |

---

## Command-line interface

`register.py` exposes a minimal CLI (installed as `register-shapes` by `pip install -e .`).
Its defaults are generic (`sigma_init=30`, `sigma_final=7`); for the setting of the paper,
pass it explicitly, or use `register_pair.py`, whose defaults are the VerSe setting:

```bash
python register.py \
    --source data/templates_vertebrae/template_7.ply \
    --target data/verse/dataset-01training/derivatives/sub-gl090/sub-gl090_dir-ax_seg-vert_msk.nii.gz \
    --target_label 7 --max_points 10000 \
    --sigma_init 10 --sigma_final 4 --normal_weight 0.05 \
    --use_fpfh --fpfh_weight 0.1 0.3 0.0 0.0 \
    --use_symmetric --trust_symmetric 0.7 \
    --solver_precision_mm 0.5 \
    --output result.ply
```

Run `python register.py --help` for all options.

---

## Batch registration from a CSV manifest

`register_batch.py` registers every row of a CSV file:

```csv
source,target,source_label,target_label,id
../data/templates_vertebrae/template_7.ply,../data/verse/.../sub-gl090_dir-ax_seg-vert_msk.nii.gz,,7,gl090_C7
```

`source` and `target` are required (relative paths are read from the manifest's folder),
the labels are needed for segmentations only, and `id` names the output folder.

```bash
python register_batch.py --manifest examples/manifest_example.csv --output_dir results/batch
```

Each row gets `<output_dir>/<id>/deformed.ply` (source deformed onto the target),
`deformed_in_source_frame.ply` (same mesh with the rigid pre-alignment undone),
`source.ply` (source as used, in its own frame), `source_prealigned.ply` (source after the
pre-alignment, in the target frame) and
`target.ply`. `<output_dir>/results.csv` lists the timings, Chamfer distance, HD95 and
log-Jacobian variance of every row. A failing row is reported there without stopping the
batch, and `--skip_existing` resumes an interrupted batch. The hyperparameters default to
the VerSe setting of the paper; run `python register_batch.py --help` for the options.

---

## Reproducing the experiments of the paper

### Vertebra templates

The repository only includes the templates used by the examples (C6, C7 and T1). The 24
templates, one per vertebra label, are archived on Zenodo ([doi:10.5281/zenodo.22940731](https://doi.org/10.5281/zenodo.22940731)):
`register_vertebrae.py` downloads the missing ones into `data/templates_vertebrae/` on its
first run (the archive is checked against its SHA-256 hash and cached, so this happens once).
To download them by hand, extract `templates_vertebrae.zip` from the Zenodo record into `data/`.

### Downloading VerSe

The repository includes a single VerSe scan (`sub-gl090`), which is enough for
`register_pair.py`, the notebook and the example manifest. **The atlas
(`register_vertebrae.py`) and batch registrations over the dataset need the VerSe'20 dataset,
which must be downloaded first.**

VerSe'20 ([github.com/anjany/verse](https://github.com/anjany/verse), CC BY-SA 4.0) comes as
one archive per split:

| Split | Archive | Size |
| ------- | --------- | ------ |
| training | `https://s3.bonescreen.de/public/VerSe-complete/dataset-verse20training.zip` | 11.5 GB |
| validation | `https://s3.bonescreen.de/public/VerSe-complete/dataset-verse20validation.zip` | 13.1 GB |
| test (used for the atlas of the paper) | `https://s3.bonescreen.de/public/VerSe-complete/dataset-verse20test.zip` | 14.0 GB |

The same data is also available on OSF ([osf.io/t98fz](https://osf.io/t98fz/)). Each archive
contains the CT scans (`rawdata/`) and the segmentations (`derivatives/`); only the
segmentations are used here, so you can extract `derivatives/` alone:

```bash
curl -O https://s3.bonescreen.de/public/VerSe-complete/dataset-verse20test.zip
unzip dataset-verse20test.zip "dataset-03test/derivatives/*" -d data/verse
rm dataset-verse20test.zip   # optional: the extracted segmentations take ~0.5 GB
```

which gives the layout expected by the scripts:

```
data/
├── templates_vertebrae/
│   └── template_<label>.ply     # one template per vertebra label (1-7: C1-C7, 8-19: T1-T12, 20-24: L1-L5)
└── verse/
    └── dataset-03test/          # dataset-01training, dataset-02validation for the other splits
        └── derivatives/
            └── sub-<id>/
                ├── sub-<id>_dir-ax_seg-vert_msk.nii.gz   # label mask
                └── sub-<id>_dir-ax_seg-subreg_ctd.json    # centroids: labels present in the scan
```

The data can live anywhere: pass its folder with `--dataset_root`.

### Notebook: one vertebra and one lung vessel tree

`notebooks/registration_examples.ipynb` runs both experiments on one case each:

- **vertebra (Fig. 2)**: the C7 template onto the C7 vertebra of `sub-gl090`, with the
  metrics of Table 1, the result, the coarse-to-fine steps and the local
  expansion/compression of the mesh;
- **lung vessel tree**: Lung250M-4B case 056 (included as
  `data/lungs/lung250m4b_case_056.npz`), with the landmark error before and after
  registration. On CPU the notebook switches to the 8k-point clouds and fewer steps, since
  the paper's setting (~30k points for this case) needs a GPU.

It runs in Jupyter (`pip install -e ".[notebook]"`, then `jupyter notebook notebooks/`) and
in Google Colab, where its first cell clones and installs the repository (use a GPU runtime).

### Vertebrae: registrations and atlases

`register_vertebrae.py` registers each template onto all the VerSe vertebrae with the same
label and reports the metrics of Table 1 for every registration. It then undoes the rigid
pre-alignment and computes, per label, the mean shape and the PCA modes of variation (the
registered meshes share the template's triangulation, so this is a per-vertex average).

It requires the VerSe'20 segmentations (see [Downloading VerSe](#downloading-verse)): with
only the scan included in the repository, there is a single shape per label and no atlas
can be computed. Registering the 978 vertebrae of the test split takes about 17 min on an
RTX 3090, and several hours on CPU.

```bash
# all 24 labels on the test split, as in the paper
python register_vertebrae.py --dataset_root data/verse --parts dataset-03test

# a single label (C7), then only recompute the atlas from the existing registrations
python register_vertebrae.py --dataset_root data/verse --parts dataset-03test --labels 7
python register_vertebrae.py --labels 7 --skip_registration --n_modes 5
```

Outputs, in `results/vertebrae/L<label>/`: the registrations with their `results.csv`
(timings and metrics), `atlas_mean.ply`, `atlas_mode<k>_{minus,plus}2sd.ply` and `atlas_pca.npz`. Registrations already
done are skipped, so an interrupted run can simply be relaunched.

### Lung vessel trees

`register_lungs.py` registers the vessel point clouds of the
[Lung250M-4B](https://github.com/multimodallearning/Lung250M-4B) test cases and reports
the target registration error on their landmarks (about 100 per case). Download the point
clouds of the test cases (`cloudsTs/`, with `coordinates/`, `distance/` and `artery_vein/`)
from the link given on that page, and the landmarks from `evaluation/lms_validation.pth`
in the same repository. The skeletonized clouds (25k to 60k points per cloud) are used, as in the paper.

```bash
python register_lungs.py --data_dir cloudsTs --landmarks lms_validation.pth
python register_lungs.py --data_dir cloudsTs --landmarks lms_validation.pth --case_id case_056 --save_vtp
```

Point clouds carry no normals, so this experiment matches points on their position, vessel
radius, vessel direction and artery/vein label (see the docstring of `register_lungs.py`);
the deformation model is the same as for the vertebrae. Outputs: `results/lungs/tre.csv` and the displacement of every
source point in `results/lungs/predictions/<case>.pth`. On an RTX 3090, the 27 test cases take ~12 min (~25 s per case, 14 to 70 s depending on
the number of points), with a mean TRE of 3.07 mm (15.45 mm before registration); on CPU,
count hours per case.

---

## Landmark-constrained registration

Passing `source_landmark_indices` and `target_landmark_indices` to `register()` adds a term in the loss that pulls corresponding vertex pairs together during optimisation. The strength of the penalty is controlled by `landmark_weight` in `RegistrationConfig`.

For hard constraints, use a `landmark_weight` of the order of 10 to 20. Landmarks are not
used in the experiments of the paper.

---

## Custom features

Correspondences are found by a nearest-neighbour search in a feature space that concatenates,
for every vertex:

```
[ xyz , normal_weight · normal , fpfh_weight · FPFH , feature_weight · custom ]
```

Any per-vertex descriptor can be added as extra channels through `source_features` and
`target_features`:

```python
import numpy as np
from optimal_steps import register, load_input, RegistrationConfig

src_mesh, _ = load_input("data/templates_vertebrae/template_7.ply")
tgt_mesh, _ = load_input("path/to/seg.nii.gz", label=7, max_points=10000)

def descriptor(mesh):          # here: mean curvature, rescaled to [0, 1]
    c = np.asarray(mesh.curvature("mean"), dtype=np.float32).reshape(-1, 1)
    lo, hi = np.percentile(c, [2, 98])
    return np.clip((c - lo) / (hi - lo), 0.0, 1.0)

result = register(
    src_mesh,
    tgt_mesh,
    config=RegistrationConfig(
        n_scales=4,
        use_fpfh=True,
        fpfh_weight=[0.1, 0.3, 0.0, 0.0],
        feature_weight=[0.2, 0.2, 0.0, 0.0],   # scalar or one value per scale
    ),
    source_features=descriptor(src_mesh),      # (n_source_points, n_channels)
    target_features=descriptor(tgt_mesh),      # (n_target_points, n_channels)
)
```

- Both sides are required, with the same number of channels. NumPy arrays and torch tensors
  both work, and a 1-D array is read as a single channel.
- The row counts must match the meshes actually passed to `register()`. Since `load_input()`
  cleans, triangulates and decimates the input, compute the descriptors on the meshes it
  returns, not on the raw files.
- `feature_weight` behaves like `fpfh_weight`: scaled by the bounding-box diagonal of the source, and settable per
  scale. FPFH descriptors are L2-normalised, so descriptors rescaled to a comparable range are
  the easiest to weight.
- Custom features are used *alongside* FPFH, not instead of it. Set `use_fpfh=False` to rely on
  your own descriptors only.
- Source descriptors are attached to the vertices and stay fixed while the source deforms
  (unlike FPFH, which is recomputed at every scale), so they must describe the shape, not the pose.

---

## Key hyperparameters

The method and its equations are described in Section 2 of the paper; the "Paper" column
gives the symbol of each parameter there.

| Parameter | Paper | Role | Recommended value |
| --------- | ----- | ---- | ----------------- |
| `sigma_init` / `sigma_final` | $r$ | Radius of the exponential kernel $k(x, y) = \exp(-\lVert x - y \rVert / r)$, from the first to the last scale (mm) | `sigma_init` ~1/10 to 1/5 of the shape diameter (the pre-alignment has already done the global work, only the large parts remain to be registered); `sigma_final` ~2 to 5× the average point spacing, smaller for finer details. Paper: 10 → 4 mm for vertebrae (70–100 mm wide, ~1 mm edges), 50 → 1.25 mm for lung trees (~270 mm wide, ~0.6 mm spacing) |
| `n_scales` | | Coarse-to-fine levels | 4 is a good default |
| `outer_steps` | | Gauss-Newton steps (matching + regularisation + deformation) per scale | 4 |
| `lambda_reg` | $\lambda$ | Regularisation strength, in $(K + \lambda L^{-1})\, p = u$ | Increase for stiffer deformations (see the note below) |
| `use_fpfh` / `fpfh_weight` | $\gamma$ | FPFH descriptors in the matching, weight per scale | Non-zero at coarse scales only, fine scales rely on geometry |
| `fpfh_radius` | | Neighbourhood radius (mm) for FPFH computation | ~2–3× the average edge length |
| `normal_weight` | | Weight of the normals in the matching | Small (0.05 for vertebrae) |
| `use_symmetric_correspondences` / `trust_symmetric` | $\kappa$ | Target position $z_i = (1-\kappa)\, y_{\sigma(i)} + \kappa\, t_i$, where $t_i$ is the barycentre of the target points whose nearest source point is $x_i$ | ~0.5–0.7: pulls the source towards the parts of the target that no forward match reaches, which reduces folding artefacts |
| `metric_type` / `metric_beta` | $L_i$ / $\alpha$ | Loss: `"plane2plane"` penalises normal displacements `metric_alpha / metric_beta` times more than sliding along the surfaces ($\alpha$ = `metric_beta / metric_alpha`) | `"plane2plane"`, `metric_alpha=1`, `metric_beta=0.1`; point clouds switch to `"point2point"` |
| `landmark_weight` | | Strength of the landmark attraction term | 0 = disabled; ~10–20 for hard constraints |
| `feature_weight` | | Weight of the custom features passed to `register()` | Same order of magnitude as `fpfh_weight` |

Two differences with the notation of the paper:

- `lambda_reg` is multiplied by $N / N_\text{ref}$, where $N$ is the number of source points
  and $N_\text{ref}$ = `LAMBDA_REG_REFERENCE_POINTS` (50 000, in `optimal_steps/config.py`), so that the
  same value gives a similar stiffness whatever the sampling: the $\lambda$ of the paper
  and `lambda_reg` are not the same number.
- `normal_weight`, `fpfh_weight` and `feature_weight` are multiplied by the length of the
  bounding-box diagonal of the source, so that they do not depend on the unit or the size of
  the data.

---

## License

Code: MIT, see [`LICENSE`](LICENSE). Please cite the MICCAI paper if you use this code.

The vertebra templates (`data/templates_vertebrae/` and the Zenodo record) are derived from
[BodyParts3D](https://dbarchive.biosciencedbc.jp/en/bodyparts3d/desc.html)
(© The Database Center for Life Science, CC BY 4.0; Mitsuhashi et al., *Nucleic Acids Res.*
2008), refined with Loop subdivision and uniform remeshing. They are distributed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Data included for the examples: one scan of [VerSe](https://github.com/anjany/verse)
(CC BY-SA 4.0) and one case of [Lung250M-4B](https://github.com/multimodallearning/Lung250M-4B)
(see its license); please cite the corresponding papers if you use them.
