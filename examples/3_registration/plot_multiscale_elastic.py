"""
Elastic metric and multiscale strategy
======================================


This examples is an implementation of the paper "Geometric Modeling in Shape Space"
by Kilian, Mitra and Pottmann. More precisely, we implement the algorithm for the
Boundary Value Problem, which is a registration between two shapes.

The framework proposed in the paper allows to find a deformation between two shapes
that minimizes an elastic metric. Optimizing directly the deformation in full
resolution and with a large number of steps usually leads to bad local minima.
To avoid this, the authors propose a multiscale strategy, where the optimization
is first performed in a coarse resolution and with a small number of steps. Then,
refinement can be done by increasing the resolution (space refinement) or the number
of steps (time refinement).

"""

import pyvista as pv
import torch

import skshapes as sks

source_color = "teal"
target_color = "red"

source = sks.PolyData("../data/elephants/pose_B.obj")
target = sks.PolyData("../data/elephants/pose_A.obj")

# Make sure that underlying simplicial complex are the same
triangles = source.triangles
target.triangles = source.triangles

plotter = pv.Plotter()
plotter.add_mesh(source.to_pyvista(), color=source_color, label="source")
plotter.add_mesh(target.to_pyvista().translate([250, 0, 0]), color=target_color, label="target")
plotter.camera_position = [
    (107.13691493781944, -436.8511227598446, 929.44474582162),
    (138.38692092895508, -5.646553039550781, 2.4097938537597656),
    (-0.07742352421458944, 0.9049853883599757, 0.41833843327279513)
    ]
plotter.add_legend()
plotter.show()

###############################################################################
# Time refinement
# --------------

lbfgs = sks.LBFGS()

def time_refinement(
        parameter: torch.Tensor,
        model: sks.BaseModel,
        loss: sks.BaseLoss,
        source: sks. PolyData,
        target: sks.PolyData,
        regularization_weight: float,
        optimizer: sks.BaseOptimizer = lbfgs,
        n_iter: int = 4,
        gpu: bool = True,
        verbose: bool = False
        ) -> tuple[torch.Tensor, sks.Registration, sks.BaseModel, list[sks.PolyData]]:
    """Double the number of steps by linear interpolation and refit the registration model.

    Parameters
    ----------
    parameter
        The parameter to refine.
    model
        The model used for the registration.
    loss
        The loss function.
    registration
        The registration object.
    source
        The source shape.
    target
        The target shape.
    regularization_weight
        The regularization weight. If None, the regularization weight of the registration is not updated.
    """

    # Copy the model
    refined_model = model.copy()

    n_steps = parameter.shape[1]

    if verbose:
        print("Doubling the number of steps by linear interpolation...")

    # Double the number of steps by linear interpolation
    n_steps = 2 * n_steps
    new_parameter = torch.zeros((parameter.shape[0], n_steps, parameter.shape[2]))
    for i in range(parameter.shape[1]):
        new_parameter[:, 2* i, :] = parameter[:, i, :] / 2
        new_parameter[:, 2 * i + 1, :] = parameter[:, i, :] / 2

    # Update the model's n_steps and the regularization weight of the registration
    if model.endpoints is not None:
        refined_model.n_steps = new_parameter.shape[1] + 1
    else:
        refined_model.n_steps = new_parameter.shape[1]

    if verbose:
        print("Optimizing the refined path wrt the metric...")

    registration = sks.Registration(
        model=refined_model,
        loss=loss,
        optimizer=optimizer,
        regularization_weight=regularization_weight,
        verbose=verbose,
        n_iter=n_iter,
        gpu=gpu,
    )

    # Fit the refined parameter
    registration.fit(source=source, target=target, initial_parameter=new_parameter)

    return registration.parameter_, refined_model

###############################################################################
# Space refinement
# ---------------

from trimesh import Trimesh
from trimesh.proximity import closest_point
from trimesh.triangles import barycentric_to_points, points_to_barycentric


@torch.no_grad
def compute_coordinates(fine, coarse):
    """Compute coordinates of the fine points in the coarse mesh.

    We follow the approach of "Geometric Modeling in Shape Space", the coordinates
    are the id of the triangle, the barycentric coordinates and the distance between
    the point and his projection in the normal direction.

    Parameters
    ----------
    fine
        The PolyData object of the fine mesh
    coarse
        The PolyData object of the coarse mesh
    """

    if not fine.n_points >= coarse.n_points:
        msg = f"The fine mesh should have more points than the coarse mesh, got {fine.n_points} and {coarse.n_points}"
        raise ValueError(msg)

    faces = coarse.triangles.numpy()
    vertices = coarse.points.numpy()

    fine_points = fine.points.numpy()

    mesh = Trimesh(vertices=vertices, faces=faces)

    closest, distance, triangle_id = closest_point(mesh=mesh, points=fine_points)

    triangle_id = torch.tensor(triangle_id, dtype=sks.int_dtype)
    closest = torch.tensor(closest, dtype=sks.float_dtype)

    assert triangle_id.shape == (fine.n_points,)
    assert closest.shape == fine.points.shape

    t = coarse.points[coarse.triangles[triangle_id]]

    barycentric = points_to_barycentric(points=closest, triangles=t)

    # descr = (triangle_id, barycentric, product with normal)

    normals = coarse.triangle_normals / coarse.triangle_normals.norm(dim=-1, keepdim=True)

    # p - p' = fine_points - closest
    a = fine.points - closest

    Ns = normals[triangle_id]

    assert a.shape == Ns.shape

    # scalar product
    orthogonal_coordinate = (a * Ns).sum(dim=-1)

    return triangle_id, barycentric, orthogonal_coordinate



@torch.no_grad
def refine(origin, coord_barycentric, triangle_id, orthogonal_coordinate):

    Ns = origin.triangle_normals[triangle_id] / origin.triangle_normals[triangle_id].norm(dim=-1, keepdim=True)

    # Get the triangle
    t = origin.points[origin.triangles[triangle_id]]

    # Get the orthogonal coordinate
    orthogonal = orthogonal_coordinate.repeat(3, 1).T * Ns

    # Get the projection
    projections = barycentric_to_points(barycentric=coord_barycentric, triangles=t)
    projections = torch.tensor(projections, dtype=sks.float_dtype)

    return projections + orthogonal

def space_refinement(
        coarse_source: sks.PolyData,
        coarse_target: sks.PolyData,
        fine_source: sks.PolyData,
        fine_target: sks.PolyData,
        coarse_model: sks.BaseModel,
        coarse_parameter: torch.Tensor,
        loss: sks.BaseLoss,
        regularization_weight: float,
        optimizer: sks.BaseOptimizer=lbfgs,
        n_iter: int=4,
        gpu: bool=True,
        verbose: bool=False
        ):


    # Compute the path at coarse level
    coarse_path = coarse_model.morph(shape=coarse_source, parameter=coarse_parameter, return_path=True).path

    # Copy the model
    fine_model = coarse_model.copy()

    if coarse_model.endpoints is not None:
        fine_model.endpoints = fine_target.points

    if verbose:
        print("Projecting the fine meshes on the coarse meshes...")

    # Compute the coordinates of the fine points in the coarse meshes
    triangle_id_source, barycentric_coord_source, orthogonal_coordinate_source = compute_coordinates(fine_source, coarse_source)
    triangle_id_target, barycentric_coord_target, orthogonal_coordinate_target = compute_coordinates(fine_target, coarse_target)


    fine_parameter = fine_model.inital_parameter(shape=fine_source)
    print(fine_parameter.shape)
    # fine_parameter = torch.zeros(size=(fine_source.n_points, fine_model.n_free_steps, 3), dtype=sks.float_dtype)

    new_points = torch.zeros_like(fine_source.points, dtype=sks.float_dtype)
    previous_points = torch.zeros_like(fine_source.points, dtype=sks.float_dtype)

    if verbose:
        print("Refining the path from coarse to fine...")

    for i, p in enumerate(coarse_path):

        previous_points = new_points

        if i == 0:
            # Force the first point to be the source
            new_points = fine_source.points

        if i == len(coarse_path) - 1 and coarse_model.endpoints is not None:
            # Force the last point to be the target
            print("ok")
            new_points = fine_target.points
            coarse_model.endpoints = fine_target.points

        else:
            newpoints_source = refine(
                origin=p,
                coord_barycentric=barycentric_coord_source,
                triangle_id=triangle_id_source,
                orthogonal_coordinate=orthogonal_coordinate_source
                )

            newpoints_target = refine(
                origin=p,
                coord_barycentric=barycentric_coord_target,
                triangle_id=triangle_id_target,
                orthogonal_coordinate=orthogonal_coordinate_target
                )

            new_points = ((i+1) / len(coarse_path)) * newpoints_source + (1 - (i+1) / len(coarse_path)) * newpoints_target

            fine_parameter[:, i-1, :] = new_points - previous_points

    if verbose:
        print("Optimizing the fine path wrt the metric...")

    registration = sks.Registration(
        model=fine_model,
        loss=loss,
        optimizer=optimizer,
        n_iter=n_iter,
        regularization_weight=regularization_weight,
        verbose=verbose,
        gpu=gpu,
    )

    registration.model = fine_model
    registration.fit(source=fine_source, target=fine_target, initial_parameter=fine_parameter)

    return registration.parameter_, fine_model


###############################################################################
# Multiscale representation
# -------------------------

n_points_coarse = 650

# Parallel decimation of source and target
decimation_module = sks.Decimation(n_points=650)
decimation_module.fit(source)
n_points = [n_points_coarse]

multisource = sks.Multiscale(source, n_points=n_points, decimation_module=decimation_module)
multitarget = sks.Multiscale(target, n_points=n_points, decimation_module=decimation_module)

coarse_source = multisource.at(n_points=n_points_coarse)
coarse_target = multitarget.at(n_points=n_points_coarse)
fine_source = multisource.at(n_points=source.n_points)
fine_target = multitarget.at(n_points=target.n_points)

plotter = pv.Plotter()
plotter.add_mesh(coarse_source.to_pyvista(), color=source_color, label="coarse source")
plotter.add_mesh(coarse_target.to_pyvista().translate([250, 0, 0]), color=target_color, label="coarse target")
plotter.camera_position = [
    (107.13691493781944, -436.8511227598446, 929.44474582162),
    (138.38692092895508, -5.646553039550781, 2.4097938537597656),
    (-0.07742352421458944, 0.9049853883599757, 0.41833843327279513)
    ]
plotter.add_legend()
plotter.show()



###############################################################################
# Linear blending
# ---------------

registration = sks.Registration(
    model=sks.IntrinsicDeformation(n_steps=10),
    loss=sks.L2Loss(),
    optimizer=sks.LBFGS(),
    n_iter=3,
    regularization_weight=0.0,
)

registration.fit(source=coarse_source, target=coarse_target)
path = registration.path_
linear_parameter = registration.parameter_

cpos = [
    (-180.28077332975926, -359.5814717933118, 468.17714455864336),
    (23.941261291503906, -56.907809257507324, 2.4097938537597656),
    (-0.06479680268614828, 0.8494856829410709, 0.5236176552025292)
    ]


plotter = pv.Plotter()
plotter.open_gif("coarse_linear.gif", fps=4)
for i in range(len(path)):
    plotter.clear_actors()
    plotter.add_mesh(source.to_pyvista(), color=source_color, opacity=0.1)
    plotter.add_mesh(target.to_pyvista(), color=target_color, opacity=0.1)
    plotter.add_mesh(path[i].to_pyvista())
    plotter.camera_position = cpos
    plotter.write_frame()
plotter.close()

###############################################################################
# Registration with `as_isometric_as_possible`
# --------------------------------------------

model = sks.IntrinsicDeformation(n_steps=10, metric="as_isometric_as_possible")
loss = sks.L2Loss()

coarse_source.stiff_edges = coarse_source.k_ring_graph(k=8)

registration = sks.Registration(
    model=model,
    loss=loss,
    optimizer=sks.LBFGS(),
    n_iter=2,
    regularization_weight=0.0001,
    verbose=True,
)

registration.fit(source=coarse_source, target=coarse_target, initial_parameter=linear_parameter)
path = registration.path_
parameter = registration.parameter_

plotter = pv.Plotter()
plotter.open_gif("coarse_isometric.gif", fps=4)
for i in range(len(path)):
    plotter.clear_actors()
    plotter.add_mesh(source.to_pyvista(), color=source_color, opacity=0.1)
    plotter.add_mesh(target.to_pyvista(), color=target_color, opacity=0.1)
    plotter.add_mesh(path[i].to_pyvista())
    plotter.camera_position = cpos
    plotter.write_frame()
plotter.close()

###############################################################################
# Time refinement
# --------------

parameter, model = time_refinement(
    parameter=parameter,
    model=model,
    loss=loss,
    source=coarse_source,
    target=coarse_target,
    regularization_weight=0.0001,
    n_iter=3,
    verbose=True
    )

path = model.morph(shape=coarse_source, parameter=parameter, return_path=True).path

plotter = pv.Plotter()
plotter.open_gif("coarse_isometric_refined.gif", fps=8)
for i in range(len(path)):
    plotter.clear_actors()
    plotter.add_mesh(source.to_pyvista(), color=source_color, opacity=0.1)
    plotter.add_mesh(target.to_pyvista(), color=target_color, opacity=0.1)
    plotter.add_mesh(path[i].to_pyvista())
    plotter.camera_position = cpos
    plotter.write_frame()
plotter.close()

###############################################################################
# Space refinement
# ---------------

parameter, model = space_refinement(
    coarse_source=coarse_source,
    coarse_target=coarse_target,
    fine_source=fine_source,
    fine_target=fine_target,
    coarse_model=model,
    loss=loss,
    coarse_parameter=parameter,
    regularization_weight=0.0001,
    n_iter=1,
    verbose=1
    )

path = model.morph(shape=fine_source, parameter=parameter, return_path=True).path

plotter = pv.Plotter()
plotter.open_gif("fine_registration.gif", fps=8)
for i in range(len(path)):
    plotter.clear_actors()
    plotter.add_mesh(source.to_pyvista(), color=source_color, opacity=0.1)
    plotter.add_mesh(target.to_pyvista(), color=target_color, opacity=0.1)
    plotter.add_mesh(path[i].to_pyvista())
    plotter.camera_position = cpos
    plotter.write_frame()
plotter.close()
