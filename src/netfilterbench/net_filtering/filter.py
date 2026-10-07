"""
Network filtering and sparsification methods.

All functions follow a unified interface:
    func(graph: nx.Graph, **kwargs) -> nx.Graph

Parameters are passed via kwargs to allow flexible integration with pipelines.

Note:
    All filters expect undirected graphs (nx.Graph). Directed graphs (nx.DiGraph)
    are automatically converted at the start of each function using
    `to_undirected()`, which collapses bidirectional edges by averaging their weights.
    The `weight_strategy` kwarg controls this behaviour (default: 'mean').
"""

import networkx as nx
import numpy as np
import random
from itertools import combinations


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def to_undirected(G: nx.DiGraph, weight_strategy: str = 'mean') -> nx.Graph:
    """
    Convert a directed graph to an undirected graph.

    When both (u, v) and (v, u) exist, their weights are combined according
    to `weight_strategy`.

    Args:
        G (nx.DiGraph): Input directed graph.
        weight_strategy (str): How to combine weights of antiparallel edges.
            Options: 'mean', 'sum', 'max', 'min', 'first'. Defaults to 'mean'.

    Returns:
        nx.Graph: Undirected graph.
    """
    strategies = {
        'mean':  lambda w1, w2: (w1 + w2) / 2,
        'sum':   lambda w1, w2: w1 + w2,
        'max':   lambda w1, w2: max(w1, w2),
        'min':   lambda w1, w2: min(w1, w2),
        'first': lambda w1, w2: w1,
    }

    if weight_strategy not in strategies:
        raise ValueError(f"weight_strategy must be one of {list(strategies.keys())}")

    combine = strategies[weight_strategy]

    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))

    for u, v, data in G.edges(data=True):
        w = data.get('weight', 1)
        if H.has_edge(u, v):
            H[u][v]['weight'] = combine(H[u][v]['weight'], w)
        else:
            H.add_edge(u, v, weight=w)

    return H


def _ensure_undirected(G: nx.Graph, kwargs: dict) -> nx.Graph:
    """Convert G to undirected if necessary, respecting weight_strategy kwarg."""
    if isinstance(G, nx.DiGraph):
        return to_undirected(G, weight_strategy=kwargs.get('weight_strategy', 'mean'))
    return G


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

def maximum_spanning_tree(G: nx.Graph, **kwargs) -> nx.Graph:
    """
    Computes the Maximum Spanning Tree (or forest if disconnected).

    Weight is treated as a similarity/importance measure (higher = stronger
    connection), consistent with the other weight-aware filters in this
    package (tmfg, threshold, disparity_filter). The tree retains the edges
    of largest total weight required to connect all nodes.

    Args:
        G (nx.Graph): Input graph.

    Returns:
        nx.Graph: Maximum spanning tree/forest.
    """
    G = _ensure_undirected(G, kwargs)
    return nx.maximum_spanning_tree(G)


def pmfg(G: nx.Graph, **kwargs) -> nx.Graph:
    """
    Computes the Planar Maximally Filtered Graph (PMFG).

    Edges are processed in descending order of weight (highest similarity/
    importance first), adding each while the graph remains planar, per
    Tumminello et al. (2005).

    Args:
        G (nx.Graph): Input graph.

    Returns:
        nx.Graph: PMFG graph.
    """
    G = _ensure_undirected(G, kwargs)

    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    edge_limit = 3 * (len(G.nodes) - 2)

    sorted_edges = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)

    for u, v, data in sorted_edges:
        H.add_edge(u, v, weight=data['weight'])

        if not nx.check_planarity(H):
            H.remove_edge(u, v)

        if H.number_of_edges() == edge_limit:
            break

    return H


def tmfg(G: nx.Graph, **kwargs) -> nx.Graph:
    """
    Computes the Triangulated Maximally Filtered Graph (TMFG).

    Args:
        G (nx.Graph): Input graph.

    Returns:
        nx.Graph: TMFG graph.
    """
    G = _ensure_undirected(G, kwargs)

    def initial_tetrahedron(weights, nodes):
        scores = {node: 0 for node in nodes}
        for (u, v), w in weights.items():
            scores[u] += w
            scores[v] += w
        return sorted(scores, key=scores.get, reverse=True)[:4]

    def compute_initial_cache(faces, V, weights):
        MaxGain, BestVertex = {}, {}
        for face in faces:
            best_gain, best_v = -np.inf, None
            for v in V:
                gain = sum(weights.get(frozenset((v, x)), 0) for x in face)
                if gain > best_gain:
                    best_gain, best_v = gain, v
            MaxGain[face] = best_gain
            BestVertex[face] = best_v
        return MaxGain, BestVertex

    def update_cache(face_removed, new_faces, V, weights, MaxGain, BestVertex, inserted_vertex):
        MaxGain.pop(face_removed, None)
        BestVertex.pop(face_removed, None)

        faces_to_update = [f for f, v in BestVertex.items() if v == inserted_vertex]

        for f in faces_to_update:
            MaxGain.pop(f, None)
            BestVertex.pop(f, None)

        faces = new_faces + faces_to_update

        for face in faces:
            best_gain, best_v = -float('inf'), None
            for v in V:
                gain = sum(weights.get(frozenset((v, x)), 0) for x in face)
                if gain > best_gain:
                    best_gain, best_v = gain, v
            MaxGain[face] = best_gain
            BestVertex[face] = best_v

    weights = {
        frozenset((u, v)): data.get('weight', 0)
        for u, v, data in G.edges(data=True)
    }

    C1 = initial_tetrahedron(weights, list(G.nodes()))
    faces = [tuple(sorted(face)) for face in combinations(C1, 3)]
    V = set(G.nodes()) - set(C1)

    P = nx.Graph()
    P.add_nodes_from(G.nodes())

    for u, v in combinations(C1, 2):
        P.add_edge(u, v, weight=weights.get(frozenset((u, v)), 0))

    MaxGain, BestVertex = compute_initial_cache(faces, V, weights)

    while V:
        face = max(MaxGain, key=MaxGain.get)
        v_new = BestVertex[face]
        V.remove(v_new)

        new_faces = []
        for a, b in combinations(face, 2):
            P.add_edge(v_new, a, weight=weights.get(frozenset((v_new, a)), 0))
            P.add_edge(v_new, b, weight=weights.get(frozenset((v_new, b)), 0))
            new_faces.append(tuple(sorted((v_new, a, b))))

        faces.remove(face)
        faces.extend(new_faces)

        update_cache(face, new_faces, V, weights, MaxGain, BestVertex, v_new)

    P.remove_edges_from([
        (u, v) for u, v, d in P.edges(data=True) if d.get('weight', 0) == 0
    ])

    return P


def threshold(G: nx.Graph, **kwargs) -> nx.Graph:
    """
    Applies a global threshold filter on edge weights.

    By default the cutoff is derived from this graph's own weight
    distribution (the `quantile`-th quantile, defaulting to the median),
    instead of a fixed absolute value: since graphs in a benchmark battery
    can have wildly different weight scales, the same fixed cutoff may keep
    almost every edge in one graph and none in another, whereas a quantile
    keeps comparable filtering strength across graphs (e.g. `quantile=0.7`
    always keeps ~30% of edges, the strongest ones, regardless of scale).
    Pass `threshold` explicitly to instead use a fixed absolute cutoff.

    Args:
        G (nx.Graph): Input graph.
        quantile (float, optional): Quantile (0-1) of this graph's edge-weight
            distribution to use as the cutoff. Defaults to 0.5 (median).
            Ignored if `threshold` is given.
        threshold (float, optional): Fixed minimum weight to keep, overriding
            `quantile` with an absolute cutoff.

    Returns:
        nx.Graph: Filtered graph.
    """
    G = _ensure_undirected(G, kwargs)

    if "threshold" in kwargs:
        t = kwargs["threshold"]
    else:
        weights = [data.get('weight', 0) for _, _, data in G.edges(data=True)]
        t = np.quantile(weights, kwargs.get("quantile", 0.5)) if weights else 0

    H = nx.Graph()
    H.add_nodes_from(G.nodes())

    for u, v, data in G.edges(data=True):
        if data.get('weight', 0) >= t:
            H.add_edge(u, v, weight=data['weight'])

    return H


def degree_based_sparsifier(G: nx.Graph, **kwargs) -> nx.Graph:
    """
    Edges are ranked according to the minimum degree of their incident
    vertices. The highest-ranked edges are retained until the desired
    target ratio is reached.

    Args:
        G (nx.Graph): Input graph.
        target_ratio (float, optional): Fraction of edges to keep. Defaults to 0.5.

    Returns:
        nx.Graph: Sparsified graph.
    """
    G = _ensure_undirected(G, kwargs)
    target_ratio = kwargs.get("target_ratio", 0.5)

    H = nx.Graph()
    H.add_nodes_from(G.nodes())

    edges = sorted(
        G.edges(data=True),
        key=lambda x: min(G.degree(x[0]), G.degree(x[1])),
        reverse=True
    )

    target_edges = int(G.number_of_edges() * target_ratio)
    H.add_edges_from(edges[:target_edges])

    return H


def random_edge_sparsifier(G: nx.Graph, **kwargs) -> nx.Graph:
    """
    Randomly removes edges.

    Args:
        G (nx.Graph): Input graph.
        target_ratio (float, optional): Fraction of edges to keep. Defaults to 0.5.
        seed (int, optional): Random seed. Defaults to 42.

    Returns:
        nx.Graph: Sparsified graph.
    """
    G = _ensure_undirected(G, kwargs)
    seed = kwargs.get("seed", 42)
    target_ratio = kwargs.get("target_ratio", 0.5)

    rng = np.random.default_rng(seed)

    H = G.copy()

    num_to_remove = int(G.number_of_edges() * (1 - target_ratio))
    edges_to_remove = list(G.edges())
    rng.shuffle(edges_to_remove)
    edges_to_remove = edges_to_remove[:num_to_remove]

    H.remove_edges_from(edges_to_remove)
    return H


def simplified_simmelian_sparsifier(G: nx.Graph, **kwargs) -> nx.Graph:
    """
    A simplified version of the method proposed by Nick et al. (2013). For
    each edge (u, v), the Simmelian strength is defined as the number of
    common neighbors shared by its endpoints, i.e. the number of triangles
    containing the edge. For each vertex, its neighbors are ranked in
    descending order by Simmelian strength, and only the edges connecting
    it to its top-`max_rank` neighbors are retained.

    Args:
        G (nx.Graph): Input graph.
        max_rank (int, optional): Number of top-ranked neighbors kept per
            vertex. Defaults to 5.

    Returns:
        nx.Graph: Sparsified graph.

    References:
        Nick, B., Lee, C., Cunningham, P., & Brandes, U. (2013). Simmelian
        Backbones: Amplifying Hidden Homophily in Facebook Networks. ASONAM.
    """
    G = _ensure_undirected(G, kwargs)
    max_rank = kwargs.get("max_rank", 5)

    def simmelian_strength(u, v):
        return len(set(G.neighbors(u)).intersection(G.neighbors(v)))

    H = nx.Graph()
    H.add_nodes_from(G.nodes())

    for u in G.nodes():
        neighbors = sorted(
            G.neighbors(u),
            key=lambda x: simmelian_strength(u, x),
            reverse=True
        )
        H.add_edges_from((u, v, G[u][v]) for v in neighbors[:max_rank])

    return H


def disparity_filter(G: nx.Graph, **kwargs) -> nx.Graph:
    """
    Implements the disparity filter technique as described in
    Serrano et al. (2009) PNAS paper.

    Uses the closed-form solution for the significance integral:
        alpha_ij = (1 - p_ij) ** (k - 1)

    Recommended alpha range according to the original paper: [0.01, 0.5].

    Args:
        G (nx.Graph): Input weighted graph.
        alpha (float, optional): Significance level for the filter. Defaults to 0.255.

    Returns:
        nx.Graph: Filtered graph.
    """
    G = _ensure_undirected(G, kwargs)
    alpha = kwargs.get("alpha", 0.255)

    H = nx.Graph()
    H.add_nodes_from(G.nodes())

    for u in G.nodes():
        k = G.degree(u)

        if k > 1:
            strength = sum(G[u][v].get('weight', 1) for v in G[u])

            for v in G[u]:
                weight = G[u][v].get('weight', 1)
                p_ij = weight / strength

                alpha_ij = (1 - p_ij) ** (k - 1)

                if alpha_ij < alpha:
                    H.add_edge(u, v, weight=weight)

    return H


def randomized_overlapping_trees(G: nx.Graph, **kwargs) -> nx.Graph:
    """
    Inspired by the Overlapping Tree Network proposed by Garas and Argyrakis
    (2009), multiple spanning trees are generated from the original network.
    At each iteration, random weights are assigned to all edges, and a
    minimum spanning tree is computed using these weights. The final network
    is the union of all generated trees, after which the original edge
    weights are restored.

    Args:
        G (nx.Graph): Input graph.
        num_trees (int, optional): Number of spanning trees to generate and combine. Defaults to 3.

    Returns:
        nx.Graph: Reduced graph.

    References:
        Garas, A., & Argyrakis, P. (2009). Filtering of complex systems
        using overlapping tree networks. EPL, arXiv:0812.3227.
    """
    G = _ensure_undirected(G, kwargs)
    num_trees = kwargs.get("num_trees", 3)

    H = nx.Graph()
    H.add_nodes_from(G.nodes())

    G_copy = G.copy()

    rng = np.random.default_rng(kwargs.get("seed", 42))

    for _ in range(num_trees):
        for u, v, d in G_copy.edges(data=True):
            d['random_weight'] = rng.random()

        T = nx.minimum_spanning_tree(G_copy, weight='random_weight')
        H.add_edges_from(T.edges(data=True))

    for u, v in H.edges():
        H[u][v]['weight'] = G[u][v].get('weight', 1)

    return H


def k_core_decomposition(G: nx.Graph, **kwargs) -> nx.Graph:
    """
    Implements the k-core decomposition network reduction technique.

    Recursively removes nodes with degree less than k until no such nodes remain.
    If k is not specified, returns the main core (largest k-core).

    Args:
        G (nx.Graph): Input graph.
        k (int, optional): The order of the core. If not specified, returns the main core.

    Returns:
        nx.Graph: k-core graph with original node set preserved.

    References:
        Batagelj, V., & Zaversnik, M. (2003). An O(m) Algorithm for Cores Decomposition of Networks.
        https://arxiv.org/abs/cs.DS/0310049
    """
    G = _ensure_undirected(G, kwargs)
    k = kwargs.get('k', None)

    core_numbers = nx.core_number(G)

    if k is None:
        k = max(core_numbers.values())

    H = G.subgraph([n for n, cn in core_numbers.items() if cn >= k])

    result = nx.Graph()
    result.add_nodes_from(G.nodes())
    result.add_edges_from(H.edges(data=True))

    return result