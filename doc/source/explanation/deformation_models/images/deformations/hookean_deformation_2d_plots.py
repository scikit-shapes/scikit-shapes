"""2D figures for the doc of hookean deformation models."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

# Young's modulus and Poisson's ratio for the material
E = 1e7
NU = 0.3

# Lamé parameters
LAMBDA = E * NU / ((1 + NU) * (1 - 2 * NU))
MU = E / (2 * (1 + NU))

# Configuration for different deformation types
DEFORMATION_CONFIG = {
    "isotropic_scaling": {
        "title": "Isotropic Scaling",
        "param_name": "s",
        "param_range_for_anim": np.linspace(0.7, 1.3, 60),
        "param_range_for_energy": np.linspace(0.6, 1.4, 100),
        "colormap_deformation": "viridis",
        "energy_color": "darkorange",
        "param_static_values": [0.7, 0.85, 1.15, 1.3],
        "x_label": "Scaling factor $s$",
        "rest_value": 1.0,
    },
    "uniaxial_stretch": {
        "title": "Uniaxial Stretch/Compression",
        "param_name": "$\\varepsilon$",
        "param_range_for_anim": np.linspace(-0.3, 0.3, 60),
        "param_range_for_energy": np.linspace(-0.4, 0.4, 100),
        "colormap_deformation": "cividis",
        "energy_color": "royalblue",
        "param_static_values": [-0.3, -0.15, 0.15, 0.3],
        "x_label": "Strain $\\varepsilon$",
        "rest_value": 0.0,
    },
    "shear": {
        "title": "Pure Shear",
        "param_name": "$\\gamma$",
        "param_range_for_anim": np.linspace(-0.4, 0.4, 60),
        "param_range_for_energy": np.linspace(-0.5, 0.5, 100),
        "colormap_deformation": "plasma",
        "energy_color": "seagreen",
        "param_static_values": [-0.4, -0.2, 0.2, 0.4],
        "x_label": "Shear $\\gamma$",
        "rest_value": 0.0,
    },
}

DPI = 300

def green_lagrange_strain(F: np.ndarray) -> np.ndarray:
    """Compute Green-Lagrange strain tensor ε = 0.5(F^T F - I)"""
    I2 = np.eye(2)
    return 0.5 * (F.T @ F - I2)


def hookean_energy(F: np.ndarray, mu: float, lam: float) -> float:
    """Compute Hookean energy: W = 0.5 * λ * (tr ε)^2 + μ * ε:ε"""
    eps = green_lagrange_strain(F)
    trace_eps = np.trace(eps)
    double_contraction = np.sum(eps * eps)  # ε:ε = Frobenius norm squared
    return 0.5 * lam * trace_eps**2 + mu * double_contraction


def compute_deformation_matrix(
    deformation_type: str, value: float
) -> np.ndarray:
    """Create deformation matrix based on type and parameter value"""
    if deformation_type == "isotropic_scaling":
        return value * np.eye(2)
    elif deformation_type == "uniaxial_stretch":
        return np.array([[1 + value, 0], [0, 1]])
    elif deformation_type == "shear":
        return np.array([[1, value], [0, 1]])
    else:
        error_msg = f"Unknown deformation type: {deformation_type}"
        raise ValueError(error_msg)


def compute_energy_values() -> tuple[list[float], list[float], list[float]]:
    """Compute energy values for different deformation types"""
    W_stretch = [
        hookean_energy(
            compute_deformation_matrix("uniaxial_stretch", e), MU, LAMBDA
        )
        for e in DEFORMATION_CONFIG["uniaxial_stretch"][
            "param_range_for_energy"
        ]
    ]
    W_shear = [
        hookean_energy(compute_deformation_matrix("shear", g), MU, LAMBDA)
        for g in DEFORMATION_CONFIG["shear"]["param_range_for_energy"]
    ]
    W_iso = [
        hookean_energy(
            compute_deformation_matrix("isotropic_scaling", s), MU, LAMBDA
        )
        for s in DEFORMATION_CONFIG["isotropic_scaling"][
            "param_range_for_energy"
        ]
    ]

    return W_stretch, W_shear, W_iso


def calculate_adaptive_limits(
    deformation_type: str, values: list[float], padding: float = 0.1
) -> tuple[float, float, float, float]:
    """Calculate adaptive plot limits based on deformation parameters."""
    # Define the square vertices
    square = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])

    # Calculate deformed coordinates for all parameter values
    all_points = []
    for val in values:
        F = compute_deformation_matrix(deformation_type, val)
        deformed_points = np.array([F @ point for point in square])
        all_points.extend(deformed_points)

    all_points = np.array(all_points)

    xmin, ymin = np.min(all_points, axis=0)
    xmax, ymax = np.max(all_points, axis=0)

    # Add padding
    width = xmax - xmin
    height = ymax - ymin
    xmin -= padding * width
    xmax += padding * width
    ymin -= padding * height
    ymax += padding * height

    return xmin, xmax, ymin, ymax


def setup_clean_figure(
    figsize: tuple[int, int] = (7, 5),
    bg_color: str = "white",
    xlim: tuple[float, float] = (-0.7, 0.7),
    ylim: tuple[float, float] = (-0.7, 0.7),
) -> tuple[plt.Figure, plt.Axes]:
    """Setup a clean figure without axes for deformation plots"""
    fig = plt.figure(figsize=figsize, facecolor=bg_color, dpi=DPI)
    ax = fig.add_subplot(111)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_axis_off()
    return fig, ax


def draw_square(
    ax: plt.Axes,
    color: str = "black",
    alpha: float = 0.3,
    linewidth: float = 1.0,
    label: str | None = None,
) -> None:
    """Draw a unit square centered at the origin"""
    square = np.array(
        [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5]]
    )
    ax.plot(
        square[:, 0],
        square[:, 1],
        color=color,
        alpha=alpha,
        linewidth=linewidth,
        linestyle="--",
        label=label,
    )


def draw_deformed_square(
    ax: plt.Axes,
    F: np.ndarray,
    color: str = "blue",
    alpha: float = 0.9,
    linewidth: float = 2.0,
    label: str | None = None,
    fill_alpha: float = 0.1,
) -> None:
    """Draw a deformed square using transformation matrix F"""
    square = np.array(
        [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5]]
    )
    deformed = np.array([F @ point for point in square])

    if fill_alpha > 0:
        polygon = Polygon(
            deformed[:-1],
            closed=True,
            alpha=fill_alpha,
            facecolor=color,
            edgecolor="none",
        )
        ax.add_patch(polygon)

    ax.plot(
        deformed[:, 0],
        deformed[:, 1],
        color=color,
        alpha=alpha,
        linewidth=linewidth,
        label=label,
    )


def draw_grid(
    ax: plt.Axes,
    F: np.ndarray,
    n: int = 5,
    color: str = "blue",
    alpha: float = 0.15,
    linewidth: float = 0.8,
) -> None:
    """Draw a grid to better visualize the deformation"""
    # Create grid lines
    for i in np.linspace(-0.5, 0.5, n + 1):
        # Horizontal line
        h_line = np.array([[-0.5, i], [0.5, i]])
        h_deformed = np.array([F @ point for point in h_line])
        ax.plot(
            h_deformed[:, 0],
            h_deformed[:, 1],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
        )

        # Vertical line
        v_line = np.array([[i, -0.5], [i, 0.5]])
        v_deformed = np.array([F @ point for point in v_line])
        ax.plot(
            v_deformed[:, 0],
            v_deformed[:, 1],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
        )


def create_static_deformation_plot(deformation_type: str) -> None:
    """Create static visualization with multiple deformation values"""
    config = DEFORMATION_CONFIG[deformation_type]

    xmin, xmax, ymin, ymax = calculate_adaptive_limits(
        deformation_type, config["param_static_values"], padding=0.15
    )

    width = xmax - xmin
    height = ymax - ymin
    aspect_ratio = width / height

    fig_width = 7
    fig_height = fig_width / aspect_ratio

    fig_height = max(4, min(8, fig_height))

    fig, ax = setup_clean_figure(
        figsize=(fig_width, fig_height), xlim=(xmin, xmax), ylim=(ymin, ymax)
    )
    draw_square(
        ax, color="black", alpha=0.7, linewidth=1.2, label="Rest state"
    )

    colors = plt.cm.get_cmap(config["colormap_deformation"])(
        np.linspace(0.15, 0.85, len(config["param_static_values"]))
    )

    for value, color in zip(
        config["param_static_values"], colors, strict=True
    ):
        F = compute_deformation_matrix(deformation_type, value)
        label = f"{config['param_name']} = {value:.2f}"
        draw_deformed_square(ax, F, color=color, label=label)
        draw_grid(ax, F, n=4, color=color)

    legend_y = -0.03 if deformation_type == "isotropic_scaling" else 0.08

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, legend_y),
        fontsize="medium",
        frameon=True,
        fancybox=True,
        framealpha=0.7,
        ncol=4,
    )
    fig.text(
        0.5,
        0.12 if deformation_type == "isotropic_scaling" else 0.14,
        config["title"],
        ha="center",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout(pad=1.5)

    return fig

    # Save as SVG
    # output_dir = Path(__file__).parent.absolute()
    # filename = f"{deformation_type}_deformation.svg"
    # output_path = output_dir / filename
    # plt.savefig(output_path)
    # print(f"Saved figure to {output_path}")


def create_energy_plot(
    deformation_type: str, energy_values: list[float]
) -> plt.Figure:
    """Create energy plot for a specific deformation type and return the figure."""
    config = DEFORMATION_CONFIG[deformation_type]
    x_range = config["param_range_for_energy"]

    fig, ax = plt.subplots(figsize=(6, 4), dpi=DPI)

    ax.plot(x_range, energy_values, color=config["energy_color"])
    ax.set_title(config["title"])
    ax.set_xlabel(config["x_label"])
    ax.set_ylabel("Energy $W(\\mathbf{F})$")
    ax.axvline(config["rest_value"], color="gray", linestyle="--", alpha=0.5)
    ax.grid(True)

    fig.tight_layout()

    return fig

    # Save as SVG
    # output_dir = Path(__file__).parent.absolute()
    # filename = f"{deformation_type}_energy.svg"
    # output_path = output_dir / filename
    # plt.savefig(output_path)
    # print(f"Saved figure to {output_path}")
    # plt.close()


def create_animation(deformation_type: str) -> None:
    """Create animation of the deformation"""
    config = DEFORMATION_CONFIG[deformation_type]
    parameter_range = config["param_range_for_anim"]
    colormap = plt.cm.get_cmap(config["colormap_deformation"])

    xmin, xmax, ymin, ymax = calculate_adaptive_limits(
        deformation_type,
        [parameter_range[0], parameter_range[-1]],
        padding=0.15,
    )

    width = xmax - xmin
    height = ymax - ymin
    aspect_ratio = width / height

    fig_width = 7
    fig_height = fig_width / aspect_ratio

    fig_height = max(4, min(8, fig_height))

    fig, ax = setup_clean_figure(
        figsize=(fig_width, fig_height), xlim=(xmin, xmax), ylim=(ymin, ymax)
    )

    def update(i: int) -> list:
        ax.clear()
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_axis_off()

        draw_square(ax, color="black", alpha=0.7, linewidth=1.2)

        value = parameter_range[i]
        F = compute_deformation_matrix(deformation_type, value)

        norm_position = (value - parameter_range[0]) / (
            parameter_range[-1] - parameter_range[0]
        )
        color_position = 0.15 + norm_position * 0.7
        color = colormap(color_position)

        fill_alpha = 0.2

        draw_deformed_square(
            ax, F, color=color, alpha=1.0, linewidth=2.5, fill_alpha=fill_alpha
        )
        draw_grid(ax, F, n=4, color=color, alpha=0.3, linewidth=1.0)

        ax.set_title(
            f"{config['title']}: {config['param_name']}={value:.2f}",
            fontsize=12,
        )
        return []

    # Create and save animation
    # anim = FuncAnimation(fig, update, frames=frames, blit=True)
    # Path(output_dir).mkdir(parents=True, exist_ok=True)
    # output_path = Path(output_dir) / f"{deformation_type}_deformation.gif"
    # anim.save(output_path, writer="pillow", fps=fps, dpi=DPI)
    # # print(f"Saved animation to {output_path}")
    # plt.close(fig)


def plot_combined(deformation_type: str) -> plt.Figure:
    """Return a figure combining static deformation (left) and energy (right) for a given deformation type."""

    config = DEFORMATION_CONFIG[deformation_type]

    fig, (ax_deformation, ax_energy) = plt.subplots(
        1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.2, 1]}, dpi=DPI
    )

    xmin, xmax, ymin, ymax = calculate_adaptive_limits(
        deformation_type, config["param_static_values"], padding=0.15
    )
    ax_deformation.set_xlim(xmin, xmax)
    ax_deformation.set_ylim(ymin, ymax)
    ax_deformation.set_aspect("equal")
    ax_deformation.set_axis_off()

    draw_square(
        ax_deformation,
        color="black",
        alpha=0.7,
        linewidth=1.2,
        label="Rest state",
    )

    colors = plt.cm.get_cmap(config["colormap_deformation"])(
        np.linspace(0.15, 0.85, len(config["param_static_values"]))
    )
    for value, color in zip(
        config["param_static_values"], colors, strict=True
    ):
        F = compute_deformation_matrix(deformation_type, value)
        label = f"{config['param_name']} = {value:.2f}"
        draw_deformed_square(ax_deformation, F, color=color, label=label)
        draw_grid(ax_deformation, F, n=4, color=color)

    legend_y = -0.05 if deformation_type == "isotropic_scaling" else 0.08
    ax_deformation.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, legend_y),
        fontsize="medium",
        frameon=True,
        fancybox=True,
        framealpha=0.7,
        ncol=4,
    )

    energy_values = [
        hookean_energy(
            compute_deformation_matrix(deformation_type, param), MU, LAMBDA
        )
        for param in config["param_range_for_energy"]
    ]
    ax_energy.plot(
        config["param_range_for_energy"],
        energy_values,
        color=config["energy_color"],
    )
    ax_energy.set_title(config["title"])
    ax_energy.set_xlabel(config["x_label"])
    ax_energy.set_ylabel("Energy $W(\\mathbf{F})$")
    ax_energy.axvline(
        config["rest_value"], color="gray", linestyle="--", alpha=0.5
    )
    ax_energy.grid(True)

    fig.suptitle(
        f"{config['title']}: Deformation and Energy",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(pad=2.0)

    return fig


def plot_isotropic_scaling_animation() -> None:
    """Create isotropic scaling animation"""
    create_animation("isotropic_scaling", frames=60, fps=24)


def plot_uniaxial_stretch_animation() -> None:
    """Create uniaxial stretch animation"""
    create_animation("uniaxial_stretch", frames=60, fps=24)


def plot_shear_animation() -> None:
    """Create shear animation"""
    create_animation("shear", frames=60, fps=24)
