import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

# ... (Keep E, NU, LAMBDA, MU, DEFORMATION_CONFIG, DPI definitions as they are) ...
E = 1e7
NU = 0.3
LAMBDA = E * NU / ((1 + NU) * (1 - 2 * NU))
MU = E / (2 * (1 + NU))
DEFORMATION_CONFIG = {  # Keep your config
    "twisting": {
        "title": "Twisting",
        "param_name": "s",
        "length_column": 2.0,
        "radius_shaft": 0.3,
        "radius_base_factor": 1.5,
        "radius_capital_factor": 1.4,
        "height_base_ratio": 0.1,
        "height_capital_ratio": 0.15,
        "n_radius": 10,
        "n_theta": 150,  # Note: This still conceptually represents resolution for a full circle
        "n_height": 50,
        "n_flutes": 75,
        "flute_depth_factor": 0.05,
        "twist_strength": -np.pi / 2,
        "color_surface": "ivory",
        "color_edges": "saddlebrown",
        "color_surface_twisted": "ivory",
        "color_edges_twisted": "saddlebrown",
        "energy_color": "darkorange",
        "param_range_for_energy": np.linspace(-1.5, 1.5, 200),
    },
    "bending": {
        "title": "Bending (Euler-Bernoulli)",
        "param_name": "F",
        "param_value_for_visualization": 100.0,
        "length_beam": 2.0,
        "height_beam": 0.1,
        "width_beam": 0.5,
        "n_length_beam": 40,
        "n_height_beam": 2,
        "n_width_beam": 10,
        "energy_color": "darkgreen",
        "param_range_for_energy": np.linspace(-200, 200, 20),
    },
}
DPI = 300


def generate_column_3d(
    length: float = DEFORMATION_CONFIG["twisting"]["length_column"],
    radius_shaft: float = DEFORMATION_CONFIG["twisting"]["radius_shaft"],
    radius_base_factor: float = DEFORMATION_CONFIG["twisting"][
        "radius_base_factor"
    ],
    radius_capital_factor: float = DEFORMATION_CONFIG["twisting"][
        "radius_capital_factor"
    ],
    height_base_ratio: float = DEFORMATION_CONFIG["twisting"][
        "height_base_ratio"
    ],
    height_capital_ratio: float = DEFORMATION_CONFIG["twisting"][
        "height_capital_ratio"
    ],
    n_radius: int = DEFORMATION_CONFIG["twisting"]["n_radius"],
    n_theta: int = DEFORMATION_CONFIG["twisting"]["n_theta"],
    n_height: int = DEFORMATION_CONFIG["twisting"]["n_height"],
    n_flutes: int = DEFORMATION_CONFIG["twisting"]["n_flutes"],
    flute_depth_factor: float = DEFORMATION_CONFIG["twisting"][
        "flute_depth_factor"
    ],
    generate_half_column: bool = True,
) -> tuple:
    """
    Generates a 3D surface of a Roman-style column.
    """
    r_base = radius_shaft * radius_base_factor
    r_capital = radius_shaft * radius_capital_factor
    h_base = length * height_base_ratio
    h_capital = length * height_capital_ratio
    capital_start_z = length - h_capital

    z_coords = np.linspace(0, length, n_height + 1)
    radius_profile = np.full_like(z_coords, radius_shaft)

    base_indices = z_coords <= h_base
    if np.any(base_indices):
        radius_profile[base_indices] = np.linspace(
            r_base, radius_shaft, np.sum(base_indices)
        )

    capital_indices = z_coords >= capital_start_z
    if np.any(capital_indices):
        radius_profile[capital_indices] = np.linspace(
            radius_shaft, r_capital, np.sum(capital_indices)
        )

    r_linspace = np.linspace(0, 1, n_radius + 1)

    if generate_half_column:
        num_theta_points_half = n_theta // 2 + 1
        theta_linspace = np.linspace(
            0, np.pi, num_theta_points_half, endpoint=True
        )
    else:
        num_theta_points_full = n_theta + 1
        theta_linspace = np.linspace(
            0, 2 * np.pi, num_theta_points_full, endpoint=True
        )

    R_norm, Theta, Z = np.meshgrid(
        r_linspace, theta_linspace, z_coords, indexing="ij"
    )

    flute_modulation = 1.0
    if n_flutes > 0 and flute_depth_factor > 0:
        base_flute_modulation = (
            1 - flute_depth_factor * (1 + np.cos(n_flutes * Theta)) / 2
        )
        flute_height_profile = np.ones_like(z_coords)
        flute_height_profile[z_coords <= h_base] = 0.0
        flute_height_profile[z_coords >= capital_start_z] = 0.0
        flute_modulation = (
            1
            - (1 - base_flute_modulation)
            * flute_height_profile[np.newaxis, np.newaxis, :]
        )

    radius_profile_reshaped = radius_profile[np.newaxis, np.newaxis, :]
    R_base = R_norm * radius_profile_reshaped
    R = R_base * (1 - (1 - flute_modulation) * R_norm)

    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)

    return X, Y, Z


def apply_torsion_3d(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    twist_strength: float = DEFORMATION_CONFIG["twisting"]["twist_strength"],
) -> tuple:
    """
    Applies a twist around the Z-axis to a 3D volume.
    """
    Z_max = np.max(Z) if np.max(Z) > 0 else 1.0

    if Z.ndim < X.ndim:
        reshape_dims = [1] * (X.ndim - 1) + [-1]
        Z_broadcast = Z.reshape(reshape_dims)
    else:
        Z_broadcast = Z

    twist_angle = twist_strength * (Z_broadcast / Z_max)
    cos_angle = np.cos(twist_angle)
    sin_angle = np.sin(twist_angle)
    X_new = X * cos_angle - Y * sin_angle
    Y_new = X * sin_angle + Y * cos_angle
    return X_new, Y_new, Z


def generate_beam(
    length: float = DEFORMATION_CONFIG["bending"]["length_beam"],
    width: float = DEFORMATION_CONFIG["bending"]["width_beam"],
    height: float = DEFORMATION_CONFIG["bending"]["height_beam"],
    n_length: int = DEFORMATION_CONFIG["bending"]["n_length_beam"],
    n_width: int = DEFORMATION_CONFIG["bending"]["n_width_beam"],
    n_height: int = DEFORMATION_CONFIG["bending"]["n_height_beam"],
) -> tuple:
    """
    Creates a 3D mesh for a rectangular beam.
    """
    x = np.linspace(0, length, n_length + 1)
    y = np.linspace(-width / 2, width / 2, n_width + 1)
    z = np.linspace(-height / 2, height / 2, n_height + 1)
    return np.meshgrid(x, y, z, indexing="ij")


def euler_bernoulli_bend_beam(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    L: float,
    F: float,
    E: float,
    I_val: float,
) -> tuple:
    """
    Applies bending to the beam geometry using the Euler-Bernoulli beam theory for a cantilever beam.

    The analytical solution for a cantilever beam with a point force F applied at the free end is:
        w(x) = (F x^2/(6EI))*(3L - x)
    and its derivative:
        w'(x) = (F x (2L - x))/(2EI),
    where:
        - w(x) is the deflection of the beam at position x,
        - L is the length of the beam,
        - E is the Young's modulus,
        - I is the moment of inertia of the beam's cross-section.

    The local rotation angle is given by theta(x) = arctan(w'(x)). The transformation is:
        x_new = x - z*sin(theta(x))
        y_new = y
        z_new = w(x) + z*cos(theta(x))
    where the undeformed beam's neutral axis is assumed to lie along z = 0.
    """
    if X.ndim != 3:
        msg = "Input arrays X, Y, Z must be 3D for bending calculation."
        raise ValueError(msg)

    xs = X[:, 0, 0]

    w = (F * xs**2 / (6 * E * I_val + 1e-12)) * (
        3 * L - xs
    )  # Add small epsilon to avoid division by zero
    wprime = (F * xs * (2 * L - xs)) / (
        2 * E * I_val + 1e-12
    )  # Add small epsilon
    theta = np.arctan(wprime)

    theta_grid = theta[:, np.newaxis, np.newaxis]
    w_grid = w[:, np.newaxis, np.newaxis]

    X_new = X - Z * np.sin(theta_grid)
    Y_new = Y
    Z_new = w_grid + Z * np.cos(theta_grid)

    return X_new, Y_new, Z_new


def green_lagrange_strain(F: np.ndarray) -> np.ndarray:
    """Compute Green-Lagrange strain tensor ε = 0.5 (F^T F - I) for a deformation gradient F."""
    I3 = np.eye(3)

    F = np.asarray(F)
    if F.ndim != 2 or F.shape != (3, 3):
        try:
            F = F.reshape(3, 3)
        except ValueError as e:
            msg = f"Input F must be reshapeable to a 3x3 matrix. Got shape {F.shape}"
            raise ValueError(msg) from e

    FT = F.T
    FTF = FT @ F
    return 0.5 * (FTF - I3)


def hookean_energy(
    F: np.ndarray, mu: float = MU, lam: float = LAMBDA
) -> float:
    """Compute Hookean energy: W = 0.5 * λ (tr ε)^2 + μ ε:ε."""
    try:
        eps = green_lagrange_strain(F)
    except ValueError as e:
        raise e

    trace_eps = np.trace(eps)
    double_contraction = np.sum(eps * eps)  # ε:ε
    return 0.5 * lam * trace_eps**2 + mu * double_contraction


def compute_total_energy(
    X_def: np.ndarray,
    Y_def: np.ndarray,
    Z_def: np.ndarray,
    X_ref: np.ndarray,
    Y_ref: np.ndarray,
    Z_ref: np.ndarray,
) -> float:
    """
    Approximates the volumetric Hookean energy over a 3D grid using finite differences
    of the deformation map φ(X_ref) = X_def.
    """
    nx, ny, nz = X_ref.shape
    if nx < 3 or ny < 3 or nz < 3:
        msg = "Grid dimensions must be at least 3x3x3 for central differences."
        raise ValueError(msg)

    dx_ref = np.gradient(X_ref, axis=0)
    dy_ref = np.gradient(Y_ref, axis=1)
    dz_ref = np.gradient(Z_ref, axis=2)

    dx = dx_ref[1:-1, 1:-1, 1:-1]
    dy = dy_ref[1:-1, 1:-1, 1:-1]
    dz = dz_ref[1:-1, 1:-1, 1:-1]

    dXdef_dXref = (X_def[2:, 1:-1, 1:-1] - X_def[:-2, 1:-1, 1:-1]) / (
        X_ref[2:, 1:-1, 1:-1] - X_ref[:-2, 1:-1, 1:-1] + 1e-12
    )
    dYdef_dXref = (Y_def[2:, 1:-1, 1:-1] - Y_def[:-2, 1:-1, 1:-1]) / (
        X_ref[2:, 1:-1, 1:-1] - X_ref[:-2, 1:-1, 1:-1] + 1e-12
    )
    dZdef_dXref = (Z_def[2:, 1:-1, 1:-1] - Z_def[:-2, 1:-1, 1:-1]) / (
        X_ref[2:, 1:-1, 1:-1] - X_ref[:-2, 1:-1, 1:-1] + 1e-12
    )

    dXdef_dYref = (X_def[1:-1, 2:, 1:-1] - X_def[1:-1, :-2, 1:-1]) / (
        Y_ref[1:-1, 2:, 1:-1] - Y_ref[1:-1, :-2, 1:-1] + 1e-12
    )
    dYdef_dYref = (Y_def[1:-1, 2:, 1:-1] - Y_def[1:-1, :-2, 1:-1]) / (
        Y_ref[1:-1, 2:, 1:-1] - Y_ref[1:-1, :-2, 1:-1] + 1e-12
    )
    dZdef_dYref = (Z_def[1:-1, 2:, 1:-1] - Z_def[1:-1, :-2, 1:-1]) / (
        Y_ref[1:-1, 2:, 1:-1] - Y_ref[1:-1, :-2, 1:-1] + 1e-12
    )

    dXdef_dZref = (X_def[1:-1, 1:-1, 2:] - X_def[1:-1, 1:-1, :-2]) / (
        Z_ref[1:-1, 1:-1, 2:] - Z_ref[1:-1, 1:-1, :-2] + 1e-12
    )
    # dYdef_dZref = (Y_def[1:-1, 1:-1, 2:] - Y_def[1:-1, 1:-1, :-2]) / (Z_ref[1:-1, 1:-1, 2:] - Z_ref[1:-1, 1:-1, :-2] + 1e-12)
    dZdef_dZref = (Z_def[1:-1, 1:-1, 2:] - Z_def[1:-1, 1:-1, :-2]) / (
        Z_ref[1:-1, 1:-1, 2:] - Z_ref[1:-1, 1:-1, :-2] + 1e-12
    )

    volumes = np.abs(dx * dy * dz)

    energy_density = np.zeros_like(dXdef_dXref)
    it = np.nditer(dXdef_dXref, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        F = np.array(
            [
                [dXdef_dXref[idx], dXdef_dYref[idx], dXdef_dZref[idx]],
                [dYdef_dXref[idx], dYdef_dYref[idx], dZdef_dYref[idx]],
                [dZdef_dXref[idx], dYdef_dYref[idx], dZdef_dZref[idx]],
            ]
        )
        try:
            energy_density[idx] = hookean_energy(F, mu=MU, lam=LAMBDA)
        except ValueError as e:
            raise e

        it.iternext()

    return np.sum(energy_density * volumes)


pv.set_plot_theme("document")


def _setup_column_visualization(
    surface, color, window_size, subtitle=None
):
    surface = surface.compute_normals(
        point_normals=True, cell_normals=False, split_vertices=True
    )

    n = surface.point_normals
    x = surface.points
    p = x - x.mean(axis=0)
    if np.mean(np.sum(p * n, axis=1)) < 0:
        surface = surface.compute_normals(flip_normals=True)

    p = pv.Plotter(window_size=window_size, notebook=False)

    if subtitle:
        p.add_text(subtitle, font_size=12)

    p.add_mesh(
        surface,
        color=color,
        smooth_shading=False,
        show_edges=False,
        pbr=True,
        metallic=0.0,
        roughness=0.8,
    )

    silhouette_width = p.shape[0] * 0.0025 * surface.length
    surface["silhouette_width"] = silhouette_width * np.ones(surface.n_points)
    silhouette = surface.warp_by_scalar(scalars="silhouette_width")

    p.add_mesh(
        silhouette,
        color="black",
        culling="front",
        interpolation="pbr",
        roughness=1,
    )

    p.enable_ssao(radius=surface.length / 16)
    p.enable_anti_aliasing("ssaa", multi_samples=32)
    p.enable_parallel_projection()

    p.remove_all_lights()
    light_intensity = 2.5
    light_elev = 40
    light_azim = 90
    headlight_intensity = 2.0

    n_lights = np.ceil(light_intensity).astype(int)

    light = pv.Light(
        light_type="camera light",
        intensity=light_intensity / n_lights,
    )
    light.set_direction_angle(light_elev, light_azim)
    for _ in range(n_lights):
        p.add_light(light)

    n_headlights = np.ceil(headlight_intensity).astype(int)
    light = pv.Light(
        light_type="headlight",
        intensity=headlight_intensity / n_headlights,
    )
    for _ in range(n_headlights):
        p.add_light(light)

    # Camera setup
    length = DEFORMATION_CONFIG["twisting"]["length_column"]
    p.camera.position = (2.5, 2.0, 2.5)
    p.camera.focal_point = (0.0, 0.0, length / 2)
    p.camera.view_up = (0, 0, 1)
    p.camera.zoom(0.8)
    p.camera.parallel_projection = True

    return p


def visualize_original_column(generate_half: bool = True):
    length = DEFORMATION_CONFIG["twisting"]["length_column"]
    radius_shaft = DEFORMATION_CONFIG["twisting"]["radius_shaft"]
    n_theta = DEFORMATION_CONFIG["twisting"]["n_theta"]
    n_flutes = DEFORMATION_CONFIG["twisting"]["n_flutes"]
    flute_depth_factor = DEFORMATION_CONFIG["twisting"]["flute_depth_factor"]

    Xc, Yc, Zc = generate_column_3d(
        length=length,
        radius_shaft=radius_shaft,
        radius_base_factor=DEFORMATION_CONFIG["twisting"][
            "radius_base_factor"
        ],
        radius_capital_factor=DEFORMATION_CONFIG["twisting"][
            "radius_capital_factor"
        ],
        height_base_ratio=DEFORMATION_CONFIG["twisting"]["height_base_ratio"],
        height_capital_ratio=DEFORMATION_CONFIG["twisting"][
            "height_capital_ratio"
        ],
        n_radius=DEFORMATION_CONFIG["twisting"]["n_radius"],
        n_theta=n_theta,
        n_height=DEFORMATION_CONFIG["twisting"]["n_height"],
        n_flutes=n_flutes,
        flute_depth_factor=flute_depth_factor,
        generate_half_column=generate_half,
    )

    grid = pv.StructuredGrid(Xc, Yc, Zc)
    surface = grid.extract_surface()
    surface = surface.clean()
    surface = surface.triangulate()

    return _setup_column_visualization(
        surface, window_size=[1000, 800], color=DEFORMATION_CONFIG["twisting"]["color_surface"]
    )


def visualize_twisted_column(generate_half: bool = True):
    length = DEFORMATION_CONFIG["twisting"]["length_column"]
    n_theta = DEFORMATION_CONFIG["twisting"]["n_theta"]
    n_flutes = DEFORMATION_CONFIG["twisting"]["n_flutes"]
    flute_depth_factor = DEFORMATION_CONFIG["twisting"]["flute_depth_factor"]
    twist_strength = DEFORMATION_CONFIG["twisting"]["twist_strength"]

    Xc, Yc, Zc = generate_column_3d(
        length=length,
        radius_shaft=DEFORMATION_CONFIG["twisting"]["radius_shaft"],
        radius_base_factor=DEFORMATION_CONFIG["twisting"][
            "radius_base_factor"
        ],
        radius_capital_factor=DEFORMATION_CONFIG["twisting"][
            "radius_capital_factor"
        ],
        height_base_ratio=DEFORMATION_CONFIG["twisting"]["height_base_ratio"],
        height_capital_ratio=DEFORMATION_CONFIG["twisting"][
            "height_capital_ratio"
        ],
        n_radius=DEFORMATION_CONFIG["twisting"]["n_radius"],
        n_theta=n_theta,
        n_height=DEFORMATION_CONFIG["twisting"]["n_height"],
        n_flutes=n_flutes,
        flute_depth_factor=flute_depth_factor,
        generate_half_column=generate_half,
    )
    Xt, Yt, Zt = apply_torsion_3d(Xc, Yc, Zc, twist_strength=twist_strength)

    grid = pv.StructuredGrid(Xt, Yt, Zt)
    surface = grid.extract_surface()
    surface = surface.clean()
    surface = surface.triangulate()

    return _setup_column_visualization(
        surface, window_size=[1000, 800], color=DEFORMATION_CONFIG["twisting"]["color_surface_twisted"]
    )


def _setup_column_subplot(
    plotter, subplot_idx, surface, color, title=None, camera_settings=None
):
    plotter.subplot(*subplot_idx)

    if title:
        plotter.add_text(title, font_size=12)

    surface = surface.compute_normals(
        point_normals=True, cell_normals=False, split_vertices=True
    )

    plotter.add_mesh(
        surface,
        color=color,
        smooth_shading=True,
        show_edges=False,
        opacity=1.0,
        pbr=True,
        metallic=0.0,
        roughness=0.8,
    )

    silhouette_width = plotter.shape[0] * 0.0025 * surface.length
    surface["silhouette_width"] = silhouette_width * np.ones(surface.n_points)
    silhouette = surface.warp_by_scalar(scalars="silhouette_width")

    plotter.add_mesh(
        silhouette,
        color="black",
        culling="front",
        interpolation="pbr",
        roughness=1,
    )

    # plotter.enable_ssao(radius=surface.length / 16) # BREAKS SUBPLOT
    # plotter.enable_anti_aliasing("ssaa", multi_samples=32) # BREAKS SUBPLOT
    plotter.enable_parallel_projection()

    plotter.remove_all_lights()
    light_intensity = 2.5
    light_elev = 40
    light_azim = 90
    headlight_intensity = 2.0

    n_lights = np.ceil(light_intensity).astype(int)

    light = pv.Light(
        light_type="camera light",
        intensity=light_intensity / n_lights,
    )
    light.set_direction_angle(light_elev, light_azim)
    for _ in range(n_lights):
        plotter.add_light(light)

    n_headlights = np.ceil(headlight_intensity).astype(int)
    light = pv.Light(
        light_type="headlight",
        intensity=headlight_intensity / n_headlights,
    )
    for _ in range(n_headlights):
        plotter.add_light(light)

    plotter.camera.position = camera_settings["position"]
    plotter.camera.focal_point = camera_settings["focal_point"]
    plotter.camera.view_up = camera_settings["view_up"]
    plotter.camera.zoom(camera_settings["zoom"])
    plotter.camera.parallel_projection = True


def side_by_side_twisting(generate_half: bool = True):
    length = DEFORMATION_CONFIG["twisting"]["length_column"]
    n_theta = DEFORMATION_CONFIG["twisting"]["n_theta"]
    n_flutes = DEFORMATION_CONFIG["twisting"]["n_flutes"]
    flute_depth_factor = DEFORMATION_CONFIG["twisting"]["flute_depth_factor"]
    twist_strength = DEFORMATION_CONFIG["twisting"]["twist_strength"]

    Xc, Yc, Zc = generate_column_3d(
        length=length,
        radius_shaft=DEFORMATION_CONFIG["twisting"]["radius_shaft"],
        radius_base_factor=DEFORMATION_CONFIG["twisting"][
            "radius_base_factor"
        ],
        radius_capital_factor=DEFORMATION_CONFIG["twisting"][
            "radius_capital_factor"
        ],
        height_base_ratio=DEFORMATION_CONFIG["twisting"]["height_base_ratio"],
        height_capital_ratio=DEFORMATION_CONFIG["twisting"][
            "height_capital_ratio"
        ],
        n_radius=DEFORMATION_CONFIG["twisting"]["n_radius"],
        n_theta=n_theta,
        n_height=DEFORMATION_CONFIG["twisting"]["n_height"],
        n_flutes=n_flutes,
        flute_depth_factor=flute_depth_factor,
        generate_half_column=generate_half,
    )
    Xt, Yt, Zt = apply_torsion_3d(Xc, Yc, Zc, twist_strength=twist_strength)

    grid_original = pv.StructuredGrid(Xc, Yc, Zc)
    surface_original = grid_original.extract_surface().clean().triangulate()

    grid_twisted = pv.StructuredGrid(Xt, Yt, Zt)
    surface_twisted = grid_twisted.extract_surface().clean().triangulate()

    p = pv.Plotter(shape=(1, 2), window_size=(1600, 800))

    camera_settings = {
        "position": (2.5, 2.0, 2.5),
        "focal_point": (0.0, 0.0, length / 2),
        "view_up": (0, 0, 1),
        "zoom": 0.8,
    }

    _setup_column_subplot(
        p,
        (0, 0),
        surface_original,
        color=DEFORMATION_CONFIG["twisting"]["color_surface"],
        title="Original",
        camera_settings=camera_settings,
    )

    _setup_column_subplot(
        p,
        (0, 1),
        surface_twisted,
        color=DEFORMATION_CONFIG["twisting"]["color_surface_twisted"],
        title="Twisted",
        camera_settings=camera_settings,
    )

    p.link_views()

    return p


def plot_energy_graphs(deformation_type: str):
    length_column = DEFORMATION_CONFIG["twisting"]["length_column"]
    n_theta_col = DEFORMATION_CONFIG["twisting"]["n_theta"]

    generate_half_for_energy = True
    Xc, Yc, Zc = generate_column_3d(
        length=length_column,
        radius_shaft=DEFORMATION_CONFIG["twisting"]["radius_shaft"],
        radius_base_factor=DEFORMATION_CONFIG["twisting"][
            "radius_base_factor"
        ],
        radius_capital_factor=DEFORMATION_CONFIG["twisting"][
            "radius_capital_factor"
        ],
        height_base_ratio=DEFORMATION_CONFIG["twisting"]["height_base_ratio"],
        height_capital_ratio=DEFORMATION_CONFIG["twisting"][
            "height_capital_ratio"
        ],
        n_radius=DEFORMATION_CONFIG["twisting"]["n_radius"],
        n_theta=n_theta_col,
        n_height=DEFORMATION_CONFIG["twisting"]["n_height"],
        n_flutes=DEFORMATION_CONFIG["twisting"]["n_flutes"],
        flute_depth_factor=DEFORMATION_CONFIG["twisting"][
            "flute_depth_factor"
        ],
        generate_half_column=generate_half_for_energy,  # Use flag
    )

    twist_vals = DEFORMATION_CONFIG["twisting"]["param_range_for_energy"]
    energy_torsion = []

    for twist in twist_vals:
        Xtc, Ytc, Ztc_twisted = apply_torsion_3d(
            Xc, Yc, Zc, twist_strength=twist
        )
        W = compute_total_energy(Xtc, Ytc, Ztc_twisted, Xc, Yc, Zc)
        energy_torsion.append(W)

    length_beam = DEFORMATION_CONFIG["bending"]["length_beam"]
    width_beam = DEFORMATION_CONFIG["bending"]["width_beam"]
    height_beam = DEFORMATION_CONFIG["bending"]["height_beam"]

    Xb, Yb, Zb = generate_beam(
        length=length_beam,
        width=width_beam,
        height=height_beam,
        n_length=DEFORMATION_CONFIG["bending"]["n_length_beam"],
        n_width=DEFORMATION_CONFIG["bending"]["n_width_beam"],
        n_height=DEFORMATION_CONFIG["bending"]["n_height_beam"],
    )

    bending_force_vals = DEFORMATION_CONFIG["bending"][
        "param_range_for_energy"
    ]
    energy_bending = []
    I_val = (width_beam * height_beam**3) / 12.0

    for force in bending_force_vals:
        Xbb, Ybb, Zbb = euler_bernoulli_bend_beam(
            Xb, Yb, Zb, L=length_beam, F=force, E=E, I_val=I_val
        )
        Wb = compute_total_energy(Xbb, Ybb, Zbb, Xb, Yb, Zb)
        energy_bending.append(Wb)

    # --- Plotting ---
    if deformation_type == "twisting":
        fig1 = plt.figure(figsize=(10, 6), dpi=DPI)
        plt.plot(
            twist_vals,
            energy_torsion,
            color=DEFORMATION_CONFIG["twisting"]["energy_color"],
            linewidth=3,
        )
        plt.title("Total Energy vs Twist (Column Torsion)", fontsize=16)
        plt.xlabel("Twist angle (rad)", fontsize=14)
        plt.ylabel("Hookean Energy", fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        return fig1
    elif deformation_type == "bending":
        fig2 = plt.figure(figsize=(10, 6), dpi=DPI)
        plt.plot(
            bending_force_vals,
            energy_bending,
            color=DEFORMATION_CONFIG["bending"]["energy_color"],
            linewidth=3,
        )
        plt.title("Total Energy vs Force (Beam Bending)", fontsize=16)
        plt.xlabel("Force", fontsize=14)
        plt.ylabel("Hookean Energy", fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        return fig2
    else:
        msg = "Invalid deformation type. Choose 'twisting' or 'bending'."
        raise ValueError(msg)


# --- Example Usage ---

# Visualize just the original half-column
# p_original_half = visualize_original_column(generate_half=True)
# p_original_half_twisted = visualize_twisted_column(generate_half=True)
# p_side_by_side = side_by_side_twisting(generate_half=True)
# p_original_half.show() # Explicit camera position
# p_original_half_twisted.show()
# p_side_by_side.show()
