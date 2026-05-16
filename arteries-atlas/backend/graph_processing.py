import numpy as np
import scipy


def laplacian_smoothing(vals: np.ndarray, adjmatrix: scipy.sparse.csr_matrix, iters: int = 5) -> np.ndarray:
    """
    Compute a graph Laplacian filter of the input array.

    Parameters
    ----------
    vals: (N, 3) np.ndarray of np.float32.
        Array of values at each node of the graph.

    adjmatrix: (N, N) scipy.sparse.csr_matrix of np.float32
        Adjacency matrix of the skeleton graph.

    iters: int
        Number of smoothing iterations to perform.

    Returns
    -------
    smoothed: (N, 3) np.ndarray of np.float32
        Smoothed values.

    """
    laplacian = scipy.sparse.csgraph.laplacian(adjmatrix)

    smoothed = vals.copy()
    for _ in range(iters):
        smoothed = smoothed - 0.1 * laplacian @ smoothed

    return smoothed.squeeze()


def graph_root(adjmatrix: scipy.sparse.csr_matrix,
               features: dict,
               radius_threshold: float = 3,
               orientation: int = 2,
               reverse: bool = False
               ) -> int:
    """
    Determine and set the root node of a vascular graph.

    The root is selected as the node with either the smallest or largest coordinate value
    (along a specified dimension) among the nodes belonging to the largest connected
    component whose radius exceeds `radius_threshold`.

    Parameters
    ----------
    adjmatrix : (N, N) scipy.sparse.csr_matrix of bool
        Adjacency matrix of the graph.

    features : dict
        Dictionary containing at least the keys ["pos", "radius"]:
        "pos": (N,3) np.ndarray of np.float32 with the 3D position of the graph nodes;
        "radius": (N,1) np.ndarray of np.float32.

    radius_threshold : float
        The minimum vessel radius for a node to be eligible as the graph root.

    orientation : int
        Index of the spatial dimension used to determine the root.

    reverse: bool
        If False, the root is the node with the smallest coordinate value in the specified dimension.
        If True, the root is the node with the largest coordinate value in the corresponding dimension.

    Returns
    -------
    root : int
        Index of the selected root node.
    """
    pos, radius = features["pos"], features["radius"]

    _, components = scipy.sparse.csgraph.connected_components(adjmatrix, directed=False)
    largest_component = np.bincount(components).argmax()
    eligible_roots = np.argwhere(np.logical_and(components == largest_component, radius > radius_threshold)).squeeze()

    if reverse:
        return eligible_roots[pos[eligible_roots, orientation].argmax()]
    else:
        return eligible_roots[pos[eligible_roots, orientation].argmin()]


def merge_components_to_root(adjmatrix: scipy.sparse.csr_matrix,
                             features: dict,
                             root: int,
                             orientation: int = 2,
                             reverse: bool = False,
                             min_component_size: int = 100,
                             max_dist_to_root: float = 50
                             ) -> scipy.sparse.csr_matrix:
    """
    Merge relevant connected components into the largest connected component of the graph.

    This operation connects additional components to the root node when they are sufficiently large, touch the image
    border, and are located close enough to the root node. It helps correct cropping artifacts that may split the
    vascular structure into multiple disconnected parts.

    Parameters
    ----------
    adjmatrix : (N, N) scipy.sparse.csr_matrix of bool
        Adjacency matrix of the graph.

    features : dict
        Dictionary containing at least the key "pos":
        "pos": (N, 3) np.ndarray of np.float32 with the 3D position of the graph nodes;

    root: int
        Index of the graph root.

    orientation : int
        Spatial dimension used to determine the border of the image that is taken into account.

    reverse: bool
        If False, the border contains the nodes with minimal coordinate values in that dimension.
        If True, the border contains the nodes with maximal coordinate values in that dimension.

    min_component_size : int
        Minimum number of nodes a component must contain to be eligible for merging.

    max_dist_to_root : float
        Maximum allowed distance between a node and the root for it to be considered for merging.

    Returns
    -------
    merged_adjmatrix : (N, N) scipy.sparse.csr_matrix of bool
        Adjacency matrix of the graph after merging.
    """
    pos = features["pos"]

    if reverse:
        border_nodes = np.argwhere(pos[:, orientation] > (pos[root, orientation] - 1)).squeeze()
    else:
        border_nodes = np.argwhere(pos[:, orientation] < (pos[root, orientation] + 1)).squeeze()

    _, components = scipy.sparse.csgraph.connected_components(adjmatrix, directed=False)
    component_sizes = np.bincount(components)

    dist_to_root = np.sqrt(np.sum(((pos[root, :] - pos[border_nodes, :]) ** 2), axis=1))

    valid_bottom_nodes = np.argwhere(np.logical_and(
        dist_to_root < max_dist_to_root,
        component_sizes[components[border_nodes]] > min_component_size
    )).squeeze()

    border_nodes = border_nodes[valid_bottom_nodes]

    row = np.concatenate([border_nodes, np.repeat(root, border_nodes.shape)])
    col = np.concatenate([np.repeat(root, border_nodes.shape), border_nodes])
    data = np.ones(shape=row.shape, dtype=bool)

    bottom_edges = scipy.sparse.csr_matrix((data, (row, col)), shape=adjmatrix.shape)

    return adjmatrix + bottom_edges


def smooth_val(val: np.ndarray, adjmatrix: scipy.sparse.csr_matrix, iters: int = 1) -> np.ndarray:
    """
    Apply a mean graph convolution to the input values.

    Parameters
    ----------
    val: (N, K) np.ndarray of np.float32
        Values to smooth.

    adjmatrix : (N, N) scipy.sparse.csr_matrix of bool
        Adjacency matrix of the graph.

    iters: int, default=1
        Number of smoothing iterations to perform.

    Returns
    -------
    smoothed: (N, K) np.ndarray of np.float32
        Smoothed values.

    """
    degrees = np.maximum(1, adjmatrix.sum(axis=1))
    smoothed = val.reshape(-1, 1)

    for _ in range(iters):
        smoothed = adjmatrix.dot(smoothed) / degrees

    return np.array(smoothed, dtype=np.float32).squeeze()


def keep_largest_component(adjmatrix: scipy.sparse.csr_matrix, features: dict, root: int) -> (
        scipy.sparse.csr_matrix, dict, int):
    """
    Return the largest connected component of the input graph.

    Parameters
    ----------
    adjmatrix : (N, N) scipy.sparse.csr_matrix of bool
        Adjacency matrix of the graph.

    features : dict
        Dictionary of (N, ...) np.ndarray containing features of the graph nodes.

    root: int
        Index of the graph root.

    Returns
    -------
    new_adjmatrix : (M, M) scipy.sparse.csr_matrix of bool
        Adjacency matrix of the new graph.

    new_features : dict
        Dictionary of (M, ...) np.ndarray containing features of the new graph nodes.

    new_root: int
        Index of the new graph root.

    """
    _, components = scipy.sparse.csgraph.connected_components(adjmatrix, directed=False)
    largest_component_mask = (components == np.bincount(components).argmax())

    new_adjmatrix = adjmatrix[largest_component_mask, :][:, largest_component_mask]
    new_features = {k: features[k][largest_component_mask] for k in features}
    new_root = np.sum(largest_component_mask[:root])

    return new_adjmatrix, new_features, new_root


def edge_weights(adjmatrix: scipy.sparse.csr_matrix, features: dict) -> scipy.sparse.csr_matrix:
    """
    Compute edges weights of the graph, used as inputs of the minimum spanning tree algorithm.

    The weight of each edge is equal to the opposite minimal radius of the edge extremities.

    Parameters
    ----------
    adjmatrix : (N, N) scipy.sparse.csr_matrix of bool
        Adjacency matrix of the graph.

    features : dict
        Dictionary containing at least the key "radius":
        "radius": (N,1) np.ndarray of np.float32.

    Returns
    -------
    weight_matrix: (N, N) scipy.sparse.csr_matrix of np.float32
        Weighted adjacency matrix of the graph.

    """
    intensity = features["intensity"].squeeze()

    adjmatrix_coo = scipy.sparse.coo_matrix(adjmatrix)
    row, col = adjmatrix_coo.row, adjmatrix_coo.col

    edge_weights = - 0.1 - (intensity[row] + intensity[col])

    data = np.array(edge_weights, dtype=np.float32)

    return scipy.sparse.coo_matrix((data, (row, col)), shape=adjmatrix.shape)


def remove_cycles(adjmatrix: scipy.sparse.csr_matrix, features: dict, root: int) -> scipy.sparse.csr_matrix:
    """
    Return a cycle-free subgraph of the input, i.e. a subtree of the initial graph.

    It computes the minimum spanning tree (MST) of the original graph, with weights equal to the opposite radius of the
    edges.

    Parameters
    ----------
    adjmatrix: (N, N) scipy.sparse.csr_matrix of bool
        Adjacency matrix of the graph.

    features: dict
        Dictionary containing at least the key "radius":
        "radius": (N,1) np.ndarray of np.float32.

    root: int
        Index of the graph root.

    Returns
    -------
    tree_adjmatrix: (N, N) scipy.sparse.csr_matrix of bool
        Adjacency matrix of the cycle-free subgraph.

    """
    weight_matrix = edge_weights(adjmatrix, features)
    mst_adjmatrix = scipy.sparse.csgraph.minimum_spanning_tree(weight_matrix).astype(bool)

    nodes, predecessors = scipy.sparse.csgraph.depth_first_order(mst_adjmatrix, i_start=root, directed=False,
                                                                 return_predecessors=True)
    row, col = predecessors[nodes][1:], nodes[1:]
    data = np.ones(shape=row.shape, dtype=bool)

    return scipy.sparse.csr_matrix((data, (row, col)), shape=mst_adjmatrix.shape)
