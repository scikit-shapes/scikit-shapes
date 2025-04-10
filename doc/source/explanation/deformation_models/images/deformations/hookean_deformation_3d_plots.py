"""
3D figures for the doc of hookean deformation models.
"""

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

# Young's modulus and Poisson's ratio for the material
E = 1e7
NU = 0.3

# Lamé parameters
LAMBDA = E * NU / ((1 + NU) * (1 - 2 * NU))
MU = E / (2 * (1 + NU))

# Configuration for different deformation types
DEFORMATION_CONFIG = {
    "twisting": {
        "title": "Twisting",
        "param_name": "s",
        "length_cylinder": 2.0,
        "radius_cylinder": 0.5,
        "n_radius_cylinder": 10,
        "n_theta_cylinder": 30,
        "n_height_cylinder": 30,
        "twist_strength": np.pi,
        "color_surface": "skyblue",
        "color_edges": "navy",
        "color_surface_twisted": "coral",
        "color_edges_twisted": "darkred",
        "energy_color": "darkorange",
        "param_range_for_energy": np.linspace(-2 * np.pi, 2 * np.pi, 20),
    },
    "bending": {
        "title": "Bending (Euler-Bernoulli)",
        "param_name": "F",  # force applied at the free end of the beam
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


def generate_cylinder_3d(
    length: float = DEFORMATION_CONFIG["twisting"]["length_cylinder"],
    radius: float = DEFORMATION_CONFIG["twisting"]["radius_cylinder"],
    n_radius: int = DEFORMATION_CONFIG["twisting"]["n_radius_cylinder"],
    n_theta: int = DEFORMATION_CONFIG["twisting"]["n_theta_cylinder"],
    n_height: int = DEFORMATION_CONFIG["twisting"]["n_height_cylinder"],
) -> tuple:
    """
    Generates a 3D volumetric cylinder, along the Z-axis, with a specified length and radius.
    """
    r = np.linspace(0, radius, n_radius + 1)  # radial
    theta = np.linspace(0, 2 * np.pi, n_theta + 1)  # angular
    z = np.linspace(0, length, n_height + 1)  # height
    R, Theta, Z = np.meshgrid(r, theta, z, indexing="ij")

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
    Z_max = np.max(Z)
    twist_angle = twist_strength * (Z / Z_max)
    X_new = X * np.cos(twist_angle) - Y * np.sin(twist_angle)
    Y_new = X * np.sin(twist_angle) + Y * np.cos(twist_angle)
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

    xs = X[:, 0, 0]
    w = (F * xs**2 / (6 * E * I_val)) * (3 * L - xs)
    wprime = (F * xs * (2 * L - xs)) / (2 * E * I_val)
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
    return 0.5 * (F.T @ F - I3)


def hookean_energy(
    F: np.ndarray, mu: float = MU, lam: float = LAMBDA
) -> float:
    """Compute Hookean energy: W = 0.5 * λ (tr ε)^2 + μ ε:ε."""
    eps = green_lagrange_strain(F)
    trace_eps = np.trace(eps)
    double_contraction = np.sum(eps * eps)
    return 0.5 * lam * trace_eps**2 + mu * double_contraction


def compute_total_energy(
    X_def: np.ndarray,
    Y_def: np.ndarray,
    Z_def: np.ndarray,
    dx: float = 1e-3,
    dy: float = 1e-3,
    dz: float = 1e-3,
) -> float:
    """
    Approximates the volumetric Hookean energy over a 3D grid using vectorized operations.
    """
    dX_dx = (X_def[2:, 1:-1, 1:-1] - X_def[:-2, 1:-1, 1:-1]) / (2 * dx)
    dY_dx = (Y_def[2:, 1:-1, 1:-1] - Y_def[:-2, 1:-1, 1:-1]) / (2 * dx)
    dZ_dx = (Z_def[2:, 1:-1, 1:-1] - Z_def[:-2, 1:-1, 1:-1]) / (2 * dx)

    dX_dy = (X_def[1:-1, 2:, 1:-1] - X_def[1:-1, :-2, 1:-1]) / (2 * dy)
    dY_dy = (Y_def[1:-1, 2:, 1:-1] - Y_def[1:-1, :-2, 1:-1]) / (2 * dy)
    dZ_dy = (Z_def[1:-1, 2:, 1:-1] - Z_def[1:-1, :-2, 1:-1]) / (2 * dy)

    dX_dz = (X_def[1:-1, 1:-1, 2:] - X_def[1:-1, 1:-1, :-2]) / (2 * dz)
    dY_dz = (Y_def[1:-1, 1:-1, 2:] - Y_def[1:-1, 1:-1, :-2]) / (2 * dz)
    dZ_dz = (Z_def[1:-1, 1:-1, 2:] - Z_def[1:-1, 1:-1, :-2]) / (2 * dz)

    n_points = np.prod(dX_dx.shape)

    dX_dx = dX_dx.flatten()
    dY_dx = dY_dx.flatten()
    dZ_dx = dZ_dx.flatten()

    dX_dy = dX_dy.flatten()
    dY_dy = dY_dy.flatten()
    dZ_dy = dZ_dy.flatten()

    dX_dz = dX_dz.flatten()
    dY_dz = dY_dz.flatten()
    dZ_dz = dZ_dz.flatten()

    energy = 0.0
    for i in range(n_points):
        dx_def = np.array([dX_dx[i], dY_dx[i], dZ_dx[i]])
        dy_def = np.array([dX_dy[i], dY_dy[i], dZ_dy[i]])
        dz_def = np.array([dX_dz[i], dY_dz[i], dZ_dz[i]])

        F_grad = np.stack([dx_def, dy_def, dz_def], axis=1)
        energy += hookean_energy(F_grad)

    return energy


pv.set_plot_theme("document")


def visualize_original_cylinder():

    radius = DEFORMATION_CONFIG["twisting"]["radius_cylinder"]
    n_radius = DEFORMATION_CONFIG["twisting"]["n_radius_cylinder"]
    n_theta = DEFORMATION_CONFIG["twisting"]["n_theta_cylinder"]
    n_height = DEFORMATION_CONFIG["twisting"]["n_height_cylinder"]
    length = DEFORMATION_CONFIG["twisting"]["length_cylinder"]

    Xc, Yc, Zc = generate_cylinder_3d(length=length, radius=radius, n_radius=n_radius, n_theta=n_theta, n_height=n_height)
    grid = pv.StructuredGrid(Xc, Yc, Zc)
    surface = grid.extract_surface()

    # Plot original cylinder
    p = pv.Plotter(window_size=[2000, 1600], notebook=False)
    # p.add_mesh(
    #     grid, style="wireframe", color="navy", line_width=1, opacity=0.5
    # )
    p.add_mesh(
        surface,
        color="skyblue",
        smooth_shading=True,
        specular=0.5,
        show_edges=True,
        edge_color="navy",
        opacity=0.5,
    )

    radius = DEFORMATION_CONFIG["twisting"]["radius_cylinder"]
    n_theta = DEFORMATION_CONFIG["twisting"]["n_theta_cylinder"]
    n_height = DEFORMATION_CONFIG["twisting"]["n_height_cylinder"]
    length = DEFORMATION_CONFIG["twisting"]["length_cylinder"]

    for i in range(0, n_theta, n_theta):  # Draw fewer lines for clarity
        theta_val = i * (2 * np.pi / n_theta)

        points = []
        for j in range(n_height + 1):
            z_val = j * (length / n_height)
            x = radius * np.cos(theta_val)
            y = radius * np.sin(theta_val)
            points.append([x, y, z_val])

        lines = []
        for j in range(n_height):
            lines.append([2, j, j+1])

        line_poly = pv.PolyData(np.array(points))
        line_poly.lines = np.hstack(lines)
        p.add_mesh(line_poly, color="blue", line_width=8)

    p.view_isometric()
    p.set_background("white")
    p.show_grid()
    p.camera.position = (2.7, 2.7, 5)
    p.camera.focal_point = (0.0, 0.0, 1.0)
    p.camera.zoom(1.0)
    p.show()


def visualize_twisted_cylinder():

    radius = DEFORMATION_CONFIG["twisting"]["radius_cylinder"]
    n_radius = DEFORMATION_CONFIG["twisting"]["n_radius_cylinder"]
    n_theta = DEFORMATION_CONFIG["twisting"]["n_theta_cylinder"]
    n_height = DEFORMATION_CONFIG["twisting"]["n_height_cylinder"]
    length = DEFORMATION_CONFIG["twisting"]["length_cylinder"]
    twist_strength = DEFORMATION_CONFIG["twisting"]["twist_strength"]

    Xc, Yc, Zc = generate_cylinder_3d(length=length, radius=radius, n_radius=n_radius, n_theta=n_theta, n_height=n_height)
    Xt, Yt, Zt = apply_torsion_3d(Xc, Yc, Zc, twist_strength=twist_strength)
    grid = pv.StructuredGrid(Xt, Yt, Zt)
    surface = grid.extract_surface()

    # Plot twisted cylinder
    p = pv.Plotter(window_size=[2000, 1600], notebook=False)
    # p.add_mesh(
    #     grid, style="wireframe", color="darkred", line_width=1, opacity=0.7
    # )
    p.add_mesh(
        surface,
        color="coral",
        smooth_shading=True,
        specular=0.5,
        show_edges=True,
        edge_color="darkred",
        opacity=0.5,
    )

    for i in range(0, n_theta, n_theta):  # Draw fewer lines for clarity
        theta_val = i * (2 * np.pi / n_theta)

        points = []
        for j in range(n_height + 1):
            z_val = j * (length / n_height)
            twist_angle = twist_strength * (z_val / length)
            x = radius * np.cos(theta_val + twist_angle)
            y = radius * np.sin(theta_val + twist_angle)
            points.append([x, y, z_val])

        lines = []
        for j in range(n_height):
            lines.append([2, j, j+1])

        line_poly = pv.PolyData(np.array(points))
        line_poly.lines = np.hstack(lines)
        p.add_mesh(line_poly, color="red", line_width=8)

    p.view_isometric()
    p.set_background("white")
    p.show_grid()
    p.camera.position = (2.7, 2.7, 5)
    p.camera.focal_point = (0.0, 0.0, 1.0)
    p.camera.zoom(1.0)
    p.show()


def visualize_original_beam():

    length = DEFORMATION_CONFIG["bending"]["length_beam"]
    width = DEFORMATION_CONFIG["bending"]["width_beam"]
    height = DEFORMATION_CONFIG["bending"]["height_beam"]
    n_length = DEFORMATION_CONFIG["bending"]["n_length_beam"]
    n_width = DEFORMATION_CONFIG["bending"]["n_width_beam"]
    n_height = DEFORMATION_CONFIG["bending"]["n_height_beam"]

    Xb, Yb, Zb = generate_beam(length=length, width=width, height=height, n_length=n_length, n_width=n_width, n_height=n_height)
    grid = pv.StructuredGrid(Xb, Yb, Zb)
    surface = grid.extract_surface()

    # Plot original beam
    p = pv.Plotter(window_size=[2000, 1600], notebook=False)
    p.add_mesh(
        surface,
        color="skyblue",
        smooth_shading=True,
        specular=0.5,
        show_edges=True,
        edge_color="navy",
        opacity=0.9,
    )
    p.view_isometric()
    p.set_background("white")
    p.show_grid()
    p.camera.position = (1.0, 1.0, 1.2)
    p.camera.focal_point = (1.0, 0.0, 0.5)
    p.camera.zoom(0.4)
    p.show()


def visualize_euler_bernoulli_bent_beam():
    """
    Visualizes a beam that is bent according to the Euler-Bernoulli beam theory (cantilever with end force F).
    """

    length = DEFORMATION_CONFIG["bending"]["length_beam"]
    width = DEFORMATION_CONFIG["bending"]["width_beam"]
    height = DEFORMATION_CONFIG["bending"]["height_beam"]
    n_length = DEFORMATION_CONFIG["bending"]["n_length_beam"]
    n_width = DEFORMATION_CONFIG["bending"]["n_width_beam"]
    n_height = DEFORMATION_CONFIG["bending"]["n_height_beam"]
    force = DEFORMATION_CONFIG["bending"]["param_value_for_visualization"]

    Xb, Yb, Zb = generate_beam(length=length, width=width, height=height, n_length=n_length, n_width=n_width, n_height=n_height)
    grid_original = pv.StructuredGrid(Xb, Yb, Zb)

    I_val = (width * height**3) / 12.0

    Xb_bent, Yb_bent, Zb_bent = euler_bernoulli_bend_beam(
        Xb,
        Yb,
        Zb,
        L=length,
        F=force,
        E=E,
        I_val=I_val,
    )
    grid_bent = pv.StructuredGrid(Xb_bent, Yb_bent, Zb_bent)

    displacement = np.linalg.norm(
        grid_bent.points - grid_original.points, axis=1
    )
    grid_bent.point_data["displacement"] = displacement

    # Plot bended beam
    p = pv.Plotter(window_size=[2000, 1600], notebook=False)
    p.add_mesh(
        grid_bent,
        scalars="displacement",
        cmap="coolwarm",
        smooth_shading=True,
        specular=0.5,
        show_edges=True,
        edge_color="grey",
        opacity=0.9,
        show_scalar_bar=True,
    )
    p.view_isometric()
    p.set_background("white")
    p.show_grid()
    p.camera.position = (1.1, 1.0, 1.2)
    p.camera.focal_point = (1.1, 0.0, 0.5)
    p.camera.zoom(0.32)
    p.show()


def plot_energy_graphs():
    radius_cylinder = DEFORMATION_CONFIG["twisting"]["radius_cylinder"]
    n_radius_cylinder = DEFORMATION_CONFIG["twisting"]["n_radius_cylinder"]
    n_theta_cylinder = DEFORMATION_CONFIG["twisting"]["n_theta_cylinder"]
    n_height_cylinder = DEFORMATION_CONFIG["twisting"]["n_height_cylinder"]
    length_cylinder = DEFORMATION_CONFIG["twisting"]["length_cylinder"]

    length_beam = DEFORMATION_CONFIG["bending"]["length_beam"]
    width_beam = DEFORMATION_CONFIG["bending"]["width_beam"]
    height_beam = DEFORMATION_CONFIG["bending"]["height_beam"]
    n_length_beam = DEFORMATION_CONFIG["bending"]["n_length_beam"]
    n_width_beam = DEFORMATION_CONFIG["bending"]["n_width_beam"]
    n_height_beam = DEFORMATION_CONFIG["bending"]["n_height_beam"]

    Xc, Yc, Zc = generate_cylinder_3d(length=length_cylinder, radius=radius_cylinder, n_radius=n_radius_cylinder, n_theta=n_theta_cylinder, n_height=n_height_cylinder)
    Xb, Yb, Zb = generate_beam(length=length_beam, width=width_beam, height=height_beam, n_length=n_length_beam, n_width=n_width_beam, n_height=n_height_beam)

    twist_vals = DEFORMATION_CONFIG["twisting"]["param_range_for_energy"]
    energy_torsion = []

    for twist in twist_vals:
        Xtc, Ytc, Ztc = apply_torsion_3d(Xc, Yc, Zc, twist_strength=twist)
        W = compute_total_energy(Xtc, Ytc, Ztc)
        energy_torsion.append(W)

    bending_force_vals = DEFORMATION_CONFIG["bending"][
        "param_range_for_energy"
    ]
    energy_bending = []

    for force in bending_force_vals:
        Xbb, Ybb, Zbb = euler_bernoulli_bend_beam(
            Xb,
            Yb,
            Zb,
            L=length_beam,
            F=force,
            E=E,
            I_val=(width_beam * height_beam**3
            )
            / 12.0,
        )
        Wb = compute_total_energy(Xbb, Ybb, Zbb)
        energy_bending.append(Wb)

    # Plot torsion energy
    plt.figure(figsize=(10, 6))
    plt.plot(twist_vals, energy_torsion, color="purple", linewidth=3)
    plt.title("Total Energy vs Twist (Torsion)", fontsize=16)
    plt.xlabel("Twist angle (rad)", fontsize=14)
    plt.ylabel("Hookean Energy", fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot bending energy
    plt.figure(figsize=(10, 6))
    plt.plot(
        bending_force_vals, energy_bending, color="darkgreen", linewidth=3
    )
    plt.title("Total Energy vs Force (Bending)", fontsize=16)
    plt.xlabel("Force", fontsize=14)
    plt.ylabel("Hookean Energy", fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Visualizations of the original and deformed configurations:
    visualize_original_cylinder()
    visualize_twisted_cylinder()
    visualize_original_beam()
    visualize_euler_bernoulli_bent_beam()  # Euler-Bernoulli beam theory (cantilever beam with point load F at the free end)

    # Plot energy graphs (based on the original Hookean energy computations)
    plot_energy_graphs()
