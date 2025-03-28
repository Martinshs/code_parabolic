import numpy as np

def a_coefficient(x):
    """
    Diffusion coefficient function a(x).

    This function defines a spatially varying diffusion coefficient:
        a(x) = 0.5 + x*(x-1)

    Parameters:
    -----------
    x : float or np.ndarray
        Spatial coordinate(s) at which to evaluate a(x).

    Returns:
    --------
    float or np.ndarray
        Diffusion coefficient evaluated at x.
    """
    return 0.5 + x*(x-1)

def b_coefficient(x):
    """
    Convection coefficient function b(x).

    This function defines the convection coefficient:
        b(x) = 0.5 * sin(pi * x)

    Parameters:
    -----------
    x : float or np.ndarray
        Spatial coordinate(s) at which to evaluate b(x).

    Returns:
    --------
    float or np.ndarray
        Convection coefficient evaluated at x.
    """
    return 0.5 * np.sin(np.pi * x)

def p_potential(x):
    """
    Potential function p(x).

    This function defines the potential term used in the reaction part of the PDE:
        p(x) = sin(pi * x)

    Parameters:
    -----------
    x : float or np.ndarray
        Spatial coordinate(s).

    Returns:
    --------
    float or np.ndarray
        Potential evaluated at x.
    """
    return np.sin(np.pi * x)

# ---------------------------------------------------------------------------
# Polynomial functions for constructing the spatial part of the exact solution.
# These are defined for each edge separately.
# ---------------------------------------------------------------------------

def y_poly(edge_id, x):
    """
    Evaluate a piecewise polynomial of degree 4 associated with a given edge.

    The polynomial is defined differently for each edge (edge_id from 1 to 10).
    These polynomials represent the spatial profile of the solution (ignoring any time factor).

    Parameters:
    -----------
    edge_id : int
        Identifier of the edge (expected values: 1 to 10).
    x : float or np.ndarray
        Spatial coordinate(s) in the reference interval [0, 1].

    Returns:
    --------
    float or np.ndarray
        Evaluated polynomial value.
    """
    if edge_id == 1:
        return 10*x**4 - 12*x**3 + x**2 + x + 1
    elif edge_id == 2:
        return -3*x**4 + x**3 + x**2 + x + 1
    elif edge_id == 3:
        return 5*x**4 - 7*x**3 + x**2 + x + 1
    elif edge_id == 4:
        return 4*x**4 - 6*x**3 + x**2 + x + 1
    elif edge_id == 5:
        return 4*x**4 - 6*x**3 + x**2 + x + 1
    elif edge_id == 6:
        return 10*x**4 - 12*x**3 + x**2 + x + 1
    elif edge_id == 7:
        return -3*x**4 + x**3 + x**2 + x + 1
    elif edge_id == 8:
        return 5*x**4 - 7*x**3 + x**2 + x + 1
    elif edge_id == 9:
        return -3*x**4 + x**3 + x**2 + x + 1
    elif edge_id == 10:
        return 4*x**4 - 6*x**3 + x**2 + x + 1
    else:
        return 0.0

def dy_poly_dx(edge_id, x):
    """
    Evaluate the first derivative of the degree-4 polynomial associated with an edge.

    This function computes the derivative of y_poly with respect to x.

    Parameters:
    -----------
    edge_id : int
        Identifier of the edge (expected values: 1 to 10).
    x : float or np.ndarray
        Spatial coordinate(s).

    Returns:
    --------
    float or np.ndarray
        The first derivative of the polynomial evaluated at x.
    """
    if edge_id == 1:
        return 40*x**3 - 36*x**2 + 2*x + 1
    elif edge_id == 2:
        return -12*x**3 + 3*x**2 + 2*x + 1
    elif edge_id == 3:
        return 20*x**3 - 21*x**2 + 2*x + 1
    elif edge_id == 4:
        return 16*x**3 - 18*x**2 + 2*x + 1
    elif edge_id == 5:
        return 16*x**3 - 18*x**2 + 2*x + 1
    elif edge_id == 6:
        return 40*x**3 - 36*x**2 + 2*x + 1
    elif edge_id == 7:
        return -12*x**3 + 3*x**2 + 2*x + 1
    elif edge_id == 8:
        return 20*x**3 - 21*x**2 + 2*x + 1
    elif edge_id == 9:
        return -12*x**3 + 3*x**2 + 2*x + 1
    elif edge_id == 10:
        return 16*x**3 - 18*x**2 + 2*x + 1
    else:
        return 0.0

def d2y_poly_dx2(edge_id, x):
    """
    Evaluate the second derivative of the degree-4 polynomial associated with an edge.

    This function computes the second derivative of y_poly with respect to x.

    Parameters:
    -----------
    edge_id : int
        Identifier of the edge (expected values: 1 to 10).
    x : float or np.ndarray
        Spatial coordinate(s).

    Returns:
    --------
    float or np.ndarray
        The second derivative of the polynomial evaluated at x.
    """
    if edge_id == 1:
        return 120*x**2 - 72*x + 2
    elif edge_id == 2:
        return -36*x**2 + 6*x + 2
    elif edge_id == 3:
        return 60*x**2 - 42*x + 2
    elif edge_id == 4:
        return 48*x**2 - 36*x + 2
    elif edge_id == 5:
        return 48*x**2 - 36*x + 2
    elif edge_id == 6:
        return 120*x**2 - 72*x + 2
    elif edge_id == 7:
        return -36*x**2 + 6*x + 2
    elif edge_id == 8:
        return 60*x**2 - 42*x + 2
    elif edge_id == 9:
        return -36*x**2 + 6*x + 2
    elif edge_id == 10:
        return 48*x**2 - 36*x + 2
    else:
        return 0.0

# ---------------------------------------------------------------------------
# Time-dependent edge solution and its derivatives.
# The exact solution is constructed as a product of a spatial polynomial and a time factor.
# ---------------------------------------------------------------------------

def y_edge(edge_id, t, x):
    """
    Evaluate the exact solution on a given edge at time t and position x.

    The solution is defined as:
        y_edge(edge_id, t, x) = y_poly(edge_id, x) * sin(2*pi*t)

    Parameters:
    -----------
    edge_id : int
        Identifier of the edge.
    t : float
        Time at which to evaluate the solution.
    x : float or np.ndarray
        Spatial coordinate(s) in [0, 1].

    Returns:
    --------
    float or np.ndarray
        Evaluated solution at (t, x).
    """
    return y_poly(edge_id, x) * np.sin(2 * np.pi * t)

def partial_t_y_edge(edge_id, t, x):
    """
    Compute the partial derivative with respect to time of the edge solution.

    This derivative is given by differentiating y_edge with respect to t.

    Parameters:
    -----------
    edge_id : int
        Identifier of the edge.
    t : float
        Time variable.
    x : float or np.ndarray
        Spatial coordinate(s).

    Returns:
    --------
    float or np.ndarray
        Partial derivative with respect to t.
    """
    return 2 * np.pi * y_poly(edge_id, x) * np.cos(2 * np.pi * t)

def partial_x_y_edge(edge_id, t, x):
    """
    Compute the partial derivative with respect to x of the edge solution.

    This is obtained by differentiating the spatial polynomial y_poly.

    Parameters:
    -----------
    edge_id : int
        Identifier of the edge.
    t : float
        Time variable.
    x : float or np.ndarray
        Spatial coordinate(s).

    Returns:
    --------
    float or np.ndarray
        Partial derivative with respect to x.
    """
    return dy_poly_dx(edge_id, x) * np.sin(2 * np.pi * t)

def d2_y_edge(edge_id, t, x):
    """
    Compute the second partial derivative with respect to x of the edge solution.

    Parameters:
    -----------
    edge_id : int
        Identifier of the edge.
    t : float
        Time variable.
    x : float or np.ndarray
        Spatial coordinate(s).

    Returns:
    --------
    float or np.ndarray
        Second derivative with respect to x.
    """
    return d2y_poly_dx2(edge_id, x) * np.sin(2 * np.pi * t)

def partial_x_a_partial_x_y_edge(edge_id, t, x):
    """
    Compute the derivative with respect to x of (a(x) * (partial_x y_edge)).

    This represents the diffusion term in the PDE, taking into account both the 
    spatial variation of a(x) and the derivative of the solution.

    Parameters:
    -----------
    edge_id : int
        Identifier of the edge.
    t : float
        Time variable.
    x : float or np.ndarray
        Spatial coordinate(s).

    Returns:
    --------
    float or np.ndarray
        The combined derivative: da/dx * (partial_x y_edge) + a(x) * (d2_y_edge).
    """
    aVal = a_coefficient(x)
    da_dx = 2 * x - 1
    return da_dx * partial_x_y_edge(edge_id, t, x) + aVal * d2_y_edge(edge_id, t, x)

def f_edge(edge_id, t, x):
    """
    Evaluate the PDE source term on an edge.

    The source term f_edge is computed as:
      f_edge = (partial_t y_edge) - (partial_x (a * partial_x y_edge)) 
               + (b * partial_x y_edge) + (p * y_edge)

    Parameters:
    -----------
    edge_id : int
        Identifier of the edge.
    t : float
        Time variable.
    x : float or np.ndarray
        Spatial coordinate(s).

    Returns:
    --------
    float or np.ndarray
        Source term evaluated at (edge_id, t, x).
    """
    val_t = partial_t_y_edge(edge_id, t, x)
    val_diff = partial_x_a_partial_x_y_edge(edge_id, t, x)
    val_conv = b_coefficient(x) * partial_x_y_edge(edge_id, t, x)
    val_pot = p_potential(x) * y_edge(edge_id, t, x)
    return val_t - val_diff + val_conv + val_pot

# ---------------------------------------------------------------------------
# Initial condition and exact solution for the PDE at time t=0 (or any t).
# ---------------------------------------------------------------------------

def initial_condition_dirichlet_full(edges, all_vertices, nx):
    """
    Construct the initial condition vector for the PDE (at t=0) with Dirichlet boundary conditions.

    The vector includes:
      - Interior edge degrees of freedom computed from y_edge(edge_id, 0, x) on a uniform grid,
      - Vertex degrees of freedom determined by evaluating y_edge at x=0 (for the start of an edge)
        or x=1 (for the end of an edge).

    Parameters:
    -----------
    edges : list of tuples
        List of edges, where each edge is a tuple (s, t).
    all_vertices : list
        Sorted list of all vertices.
    nx : int
        Number of interior degrees of freedom per edge.

    Returns:
    --------
    np.ndarray
        The initial condition vector (of size E*nx + len(all_vertices)).
    """
    E = len(edges)
    n_vts = len(all_vertices)
    total_dof = E * nx + n_vts
    y0 = np.zeros(total_dof)
    # (A) Set initial values for interior edge dofs.
    dx = 1.0 / (nx + 1)
    for i_edge in range(E):
        edge_offset = i_edge * nx
        edge_id = i_edge + 1
        # Create a uniform grid on the edge (excluding endpoints).
        x_nodes = np.linspace(dx, 1.0 - dx, nx)
        for k2 in range(nx):
            x_val = x_nodes[k2]
            y0[edge_offset + k2] = y_edge(edge_id, 0, x_val)

    # (B) Set initial values for vertex dofs.
    # Build an adjacency list to determine whether a vertex is at the start (x=0)
    # or the end (x=1) of an edge.
    max_v = max(all_vertices)
    adjacency_list = [[] for _ in range(max_v + 1)]
    for e_idx, (s, t) in enumerate(edges):
        adjacency_list[s].append((e_idx, 's'))  # 's' indicates start of edge.
        adjacency_list[t].append((e_idx, 't'))  # 't' indicates end of edge.

    def vertex_dof_index(v):
        """
        Compute the global degree of freedom index for vertex v.

        The index is offset by the total number of edge dofs (E*nx) plus the vertex's position in all_vertices.
        """
        return E * nx + all_vertices.index(v)

    for v in all_vertices:
        v_dof = vertex_dof_index(v)
        # If the vertex is connected to any edge, use the first incident edge to define its initial value.
        if adjacency_list[v]:
            (e_idx, pos) = adjacency_list[v][0]  # Use the first edge.
            edge_id = e_idx + 1
            if pos == 's':
                init_val = y_edge(edge_id, 0, 0.0)
            else:
                init_val = y_edge(edge_id, 0, 1.0)
        else:
            # This case should not occur; default to 0.
            init_val = 0.0
        y0[v_dof] = init_val  # Exact solution at t=0 for vertex v.
    return y0

def compute_exact_solution(edges, all_vertices, interior_vertices, boundary_vertices, nx, t):
    """
    Compute the exact solution vector at time t.

    The exact solution is assembled for:
      - Interior edge dofs computed via y_edge(edge_id, t, x) on a uniform grid,
      - Vertex dofs determined by evaluating y_edge at x=0 (if the vertex is a start)
        or x=1 (if the vertex is an end).

    Parameters:
    -----------
    edges : list of tuples
        List of edges, where each edge is given as (s, t).
    all_vertices : list
        Sorted list of all vertices.
    interior_vertices : list
        List of interior vertices (not used explicitly in this function).
    boundary_vertices : list
        List of boundary vertices (not used explicitly in this function).
    nx : int
        Number of interior degrees of freedom per edge.
    t : float
        Time at which to compute the exact solution.

    Returns:
    --------
    np.ndarray
        The exact solution vector of size (E*nx + len(all_vertices)).
    """
    E = len(edges)
    total_dof = E * nx + len(all_vertices)
    y_ex = np.zeros(total_dof)

    dx = 1.0 / (nx + 1)
    # Compute the solution on the interior edge nodes.
    for i_edge in range(E):
        offset = i_edge * nx
        edge_id = i_edge + 1
        x_nodes = np.linspace(dx, 1.0 - dx, nx)
        for k2 in range(nx):
            x_val = x_nodes[k2]
            y_ex[offset + k2] = y_edge(edge_id, t, x_val)

    # Compute the solution at the vertices.
    # Here, for each vertex, check each edge to see if the vertex corresponds to the start (x=0)
    # or the end (x=1) and use the corresponding value from y_edge.
    for v in all_vertices:
        for i, (iv, fv) in enumerate(edges):
            if v == iv:
                v_dof = E * nx + v
                y_ex[v_dof] = y_edge(i + 1, t, 0)
            elif v == fv:
                v_dof = E * nx + v
                y_ex[v_dof] = y_edge(i + 1, t, 1)
    return y_ex

# ---------------------------------------------------------------------------
# Auxiliary functions for visualization and boundary conditions.
# ---------------------------------------------------------------------------

def define_positions_example_1():
    """
    Define spatial positions for vertices for visualization purposes.

    Returns:
    --------
    dict
        A dictionary mapping vertex identifiers (e.g., 'v1', 'v2', ...) to 2D coordinates.
    """
    positions = {
        'v1': [-1.9, 7.95],
        'v2': [-1.9, 2.05],
        'v3': [1.3, 5],
        'v4': [6, 5],
        'v5': [9.5, 7.95],
        'v6': [10, 2.05],
        'v7': [14, 5],
        'v8': [18.7, 5],
        'v9': [21.9, 7.95],
        'v10': [21.9, 2.05]
    }
    return positions

def g_boundary_condition_ex1(t, v):
    """
    Define the Dirichlet boundary condition for the PDE.

    The boundary condition is given by:
        g(t, v) = sin(2*pi*t)
    for any vertex v.

    Parameters:
    -----------
    t : float
        Time variable.
    v : any
        Vertex identifier (unused in the computation).

    Returns:
    --------
    float
        The boundary value at time t.
    """
    return np.sin(2 * np.pi * t)
