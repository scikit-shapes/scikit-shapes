import numpy as np
import scipy


def laplacian_smoothing(vals: np.ndarray, adjmatrix: scipy.sparse.csr_matrix, iters: int = 5) -> np.ndarray:
    """
    Compute a graph Laplacian filter of the input array.

    Parameters
    ----------
    vals: (N, 3) np.ndarray of np.float32.
        Array of values at each node of the graph.

    adjmatrix : (N, N) scipy.sparse.csr_matrix of np.float32
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


def graph_root(adjmatrix: scipy.sparse.csr_matrix, features: dict) -> int:
    """
    Set the root of the vascular graph.

    The chosest root is the lowest node (in terms of Z position) among the nodes of the largest connected component with
    radius larger than 3.

    Parameters
    ----------
    adjmatrix : (N, N) scipy.sparse.csr_matrix of bool
        Adjacency matrix of the graph.

    features : dict
        Dictionary containing at least the keys ["pos", "radius"]:
        "pos": (N,3) np.ndarray of np.float32 with the 3D position of the graph nodes;
        "radius": (N,1) np.ndarray of np.float32.

    Returns
    -------
    root: int
        Index of the graph root.

    """
    pos, radius = features["pos"], features["radius"]

    _, components = scipy.sparse.csgraph.connected_components(adjmatrix, directed=False)
    largest_component = np.bincount(components).argmax()
    eligible_roots = np.argwhere(np.logical_and(components == largest_component, radius > 3)).squeeze()

    return eligible_roots[pos[eligible_roots, 2].argmin()]


def merge_components_to_root(adjmatrix: scipy.sparse.csr_matrix, features: dict, root: int) -> scipy.sparse.csr_matrix:
    """
    Merge the meaning connected components to the largest connected component of the graph.

    Nodes are linked to the root as soon as their Z position is lower than the root and their euclidean distance to the
    root is lower than 50.

    Parameters
    ----------
    adjmatrix : (N, N) scipy.sparse.csr_matrix of bool
        Adjacency matrix of the graph.

    features : dict
        Dictionary containing at least the key "pos":
        "pos": (N, 3) np.ndarray of np.float32 with the 3D position of the graph nodes;

    root: int
        Index of the graph root.

    Returns
    -------
    merged_adjmatrix: (N, N) scipy.sparse.csr_matrix of bool
        Adjacency matrix of the merged graph.

    """
    pos = features["pos"]

    bottom_nodes = np.argwhere(pos[:, 2] < (pos[root, 2] + 1)).squeeze()

    _, components = scipy.sparse.csgraph.connected_components(adjmatrix, directed=False)
    component_sizes = np.bincount(components)

    dist_to_root = np.sqrt(np.sum(((pos[root, :] - pos[bottom_nodes, :]) ** 2), axis=1))

    valid_bottom_nodes = np.argwhere(np.logical_and(
        dist_to_root < 50,
        component_sizes[components[bottom_nodes]] > 100
    )).squeeze()

    bottom_nodes = bottom_nodes[valid_bottom_nodes]

    row = np.concatenate([bottom_nodes, np.repeat(root, bottom_nodes.shape)])
    col = np.concatenate([np.repeat(root, bottom_nodes.shape), bottom_nodes])
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
    # degrees = np.maximum(1, np.array(adjmatrix.sum(axis=1))).squeeze()
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
    # TODO améliorer critère
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
