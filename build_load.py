import numpy as np

##############################################################################
# PDE Load Vector Construction for Edge and Vertex Degrees of Freedom
##############################################################################

def compute_load_vector(f_edge, edges, nx, all_vertices, t):
    """
    Construct the global load vector for the PDE source term on a 1D graph.

    The source term is given by f_edge(edge_id, t, x) on each edge.
    The global load vector is assembled by considering:
      - 'nx' interior degrees of freedom per edge,
      - one degree of freedom per vertex in 'all_vertices'.

    The resulting vector F has size (E * nx + number_of_vertices), where E is 
    the number of edges. Integration is performed using a 2-point Gaussian quadrature.

    Parameters:
    -----------
    f_edge : callable
        Function representing the source term on an edge. It should have the signature
        f_edge(edge_id, t, x), where edge_id is an integer, t is time, and x is the spatial coordinate.
    edges : list of tuples
        List of edges, with each edge represented as a tuple (s, t2) of vertex indices.
    nx : int
        Number of interior degrees of freedom per edge.
    all_vertices : list
        Sorted list of all vertices in the graph.
    t : float
        Current time value for which the load is being evaluated.

    Returns:
    --------
    np.ndarray
        Global load vector of size (E * nx + len(all_vertices)).
    """
    # Number of edges in the graph.
    E = len(edges)
    # Total degrees of freedom: interior dofs on edges plus one dof per vertex.
    total_dof = E * nx + len(all_vertices)

    # Initialize the global load vector F with zeros.
    F = np.zeros(total_dof)

    # Define 2-point Gaussian quadrature points and weights for integration.
    gauss_pts = np.array([-1/np.sqrt(3), +1/np.sqrt(3)])
    gauss_wts = np.array([1.0, 1.0])

    # Compute the uniform sub-interval length for an edge parameterized on [0, 1].
    # There are (nx+1) sub-intervals in total.
    dx = 1.0 / (nx + 1)

    ##########################################################################
    # A) Compute Load Contributions for Interior Edge Degrees of Freedom
    ##########################################################################

    def interior_index(k):
        """
        Map sub-interval index k to the corresponding interior degree of freedom index.

        Parameters:
        -----------
        k : int
            Sub-interval index.

        Returns:
        --------
        int or None
            If 1 <= k <= nx, returns the local dof index (k-1);
            Otherwise, returns None (indicating a boundary node with no interior dof).
        """
        if 1 <= k <= nx:
            return k - 1
        return None

    # Loop over each edge.
    for i_edge, (s, t2) in enumerate(edges):
        # Compute the global offset for the dofs corresponding to this edge.
        offset = i_edge * nx
        # The PDE source is associated with edge_id = i_edge + 1.
        edge_id = i_edge + 1

        # Subdivide the edge [0,1] into (nx+1) sub-intervals.
        for i_sub in range(nx + 1):
            xL = i_sub * dx         # Left endpoint of the sub-interval.
            xR = (i_sub + 1) * dx     # Right endpoint of the sub-interval.
            # Map sub-interval endpoints to local dof indices.
            dofL = interior_index(i_sub)
            dofR = interior_index(i_sub + 1)

            # Skip sub-intervals that lie completely on the boundaries.
            if (dofL is None) and (dofR is None):
                continue

            # Compute mid-point and half-length of the sub-interval for affine mapping.
            mid = 0.5 * (xL + xR)
            half = 0.5 * (xR - xL)

            # Perform 2-point Gaussian quadrature on the sub-interval.
            for (gp, gw) in zip(gauss_pts, gauss_wts):
                # Map the quadrature point from [-1,1] to [xL,xR].
                xx = mid + half * gp
                w = half * gw   # Scale the quadrature weight.

                # Evaluate the source term f_edge at the physical point.
                val_f = f_edge(edge_id, t, xx)

                # Compute contributions using the linear basis functions.
                if dofL is not None:
                    # Basis function associated with the left node: phiL(x) = (xR - x)/(xR - xL)
                    phiL = (xR - xx) / (xR - xL)
                    F[offset + dofL] += w * (val_f * phiL)
                if dofR is not None:
                    # Basis function associated with the right node: phiR(x) = (xx - xL)/(xR - xL)
                    phiR = (xx - xL) / (xR - xL)
                    F[offset + dofR] += w * (val_f * phiR)

    ##########################################################################
    # B) Compute Load Contributions for Vertex Degrees of Freedom ("Tripod" Integrals)
    ##########################################################################

    # Build an adjacency list: for each vertex v, store the indices of the edges that touch v.
    max_v = max(all_vertices)
    adjacency_list = [[] for _ in range(max_v + 1)]
    for e_idx, (s, t2) in enumerate(edges):
        adjacency_list[s].append(e_idx)
        adjacency_list[t2].append(e_idx)

    # Create a mapping from vertex to its index in the sorted all_vertices list.
    v2index = {v: i for (i, v) in enumerate(all_vertices)}
    def vertex_dof_index(v):
        """
        Compute the global degree of freedom index for vertex v.

        The index is offset by E * nx (all edge dofs) plus the position of v in all_vertices.

        Parameters:
        -----------
        v : int or hashable
            Vertex identifier.

        Returns:
        --------
        int
            Global dof index corresponding to vertex v.
        """
        return E * nx + v2index[v]

    # Loop over each vertex and integrate the PDE source over the small boundary sub-intervals.
    for v in all_vertices:
        v_dof = vertex_dof_index(v)
        load_v = 0  # Local accumulator for contributions at vertex v.

        # Process each edge incident on vertex v.
        for e_idx in adjacency_list[v]:
            (sv, tv) = edges[e_idx]
            edge_id = e_idx + 1

            if sv == v:
                # Vertex v is the starting vertex of the edge (parametric x = 0),
                # so integrate over the sub-interval [0, dx] with shape function phi_v(x) = 1 - x/dx.
                mid = 0.5 * dx
                half = 0.5 * dx
                for (gp, gw) in zip(gauss_pts, gauss_wts):
                    xx = mid + half * gp   # Map quadrature point to [0, dx].
                    w = half * gw
                    phi_v = 1.0 - xx / dx
                    val_f = f_edge(edge_id, t, xx)
                    load_v += w * (val_f * phi_v)
            elif tv == v:
                # Vertex v is the ending vertex of the edge (parametric x = 1),
                # so integrate over the sub-interval [1-dx, 1] with shape function phi_v(x) = (x - (1-dx))/dx.
                xL = 1.0 - dx
                xR = 1.0
                mid = 0.5 * (xL + xR)
                half = 0.5 * (xR - xL)
                for (gp, gw) in zip(gauss_pts, gauss_wts):
                    xx = mid + half * gp  # Map quadrature point to [1-dx, 1].
                    w = half * gw
                    phi_v = (xx - xL) / dx
                    val_f = f_edge(edge_id, t, xx)
                    load_v += w * (val_f * phi_v)

        # Add the accumulated load for vertex v to its global dof.
        F[v_dof] += load_v

    # Return the assembled global load vector.
    return F
