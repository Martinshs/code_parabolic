import numpy as np

#############################################################################
#    Build M, A, B, C for edges + "tripod" for ALL vertices.
#    Both boundary and interior vertices get degrees of freedom (dofs).
#    First, we build local edge matrices, then add "tripod" integrals for each vertex
#    that touches an edge endpoint.
##############################################################################

def build_local_edge_matrices_all(nx, a_func, b_func, p_func):
    """
    Compute the local finite element matrices on an edge using Gaussian quadrature.

    The function computes four matrices:
      - M_loc: Mass matrix, corresponding to the integral of phi_i * phi_j.
      - A_loc: Stiffness matrix weighted by the coefficient a(x), corresponding to the integral of a(x) * dphi_i * dphi_j.
      - B_loc: Potential matrix weighted by the potential function p(x), corresponding to the integral of p(x) * phi_i * phi_j.
      - C_loc: Convection matrix weighted by the coefficient b(x), corresponding to the integral of b(x) * phi_i * dphi_j.

    The integration is performed on subintervals of the edge using a two-point Gaussian quadrature.
    A simple linear Lagrange basis is used on each subinterval.

    Parameters:
    -----------
    nx : int
        Number of interior discretization points along the edge (number of local degrees of freedom).
    a_func : callable
        Coefficient function a(x) for the diffusion term.
    b_func : callable
        Coefficient function b(x) for the convection term.
    p_func : callable
        Potential function p(x) for the reaction term.

    Returns:
    --------
    tuple of np.ndarray
        A tuple (M_loc, A_loc, B_loc, C_loc) of shape (nx, nx) representing the local matrices.
    """
    M_loc = np.zeros((nx, nx))
    A_loc = np.zeros((nx, nx))
    B_loc = np.zeros((nx, nx))
    C_loc = np.zeros((nx, nx))

    # Spatial discretization step on the reference interval [0, 1]
    dx = 1.0 / (nx + 1)
    # Two-point Gaussian quadrature points and weights on the reference element [-1,1]
    gauss_pts = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
    gauss_wts = np.array([1.0, 1.0])

    def interior_index(k):
        # Map the global subinterval index to a local degree of freedom index
        if 1 <= k <= nx:
            return k - 1
        return None

    # Loop over all subintervals in the partition of the edge.
    for i_sub in range(nx + 1):
        xL = i_sub * dx           # Left endpoint of subinterval
        xR = (i_sub + 1) * dx       # Right endpoint of subinterval
        dofL = interior_index(i_sub)
        dofR = interior_index(i_sub + 1)
        # If both endpoints do not correspond to any local dof, skip the subinterval.
        if (dofL is None) and (dofR is None):
            continue
        # Compute mid-point and half-length of the subinterval (for affine mapping).
        mid = 0.5 * (xL + xR)
        half = 0.5 * (xR - xL)
        # Loop over Gaussian quadrature points.
        for (gp, gw) in zip(gauss_pts, gauss_wts):
            xx = mid + half * gp  # Map quadrature point to the physical subinterval.
            w = half * gw         # Adjust weight for the subinterval length.
            a_val = a_func(xx)
            b_val = b_func(xx)
            p_val = p_func(xx)

            # Evaluate basis functions and their derivatives.
            phiL = 0; dphiL = 0
            if dofL is not None:
                # Linear basis function on the left node of the subinterval.
                phiL = (xR - xx) / (xR - xL)
                dphiL = -1.0 / (xR - xL)
            phiR = 0; dphiR = 0
            if dofR is not None:
                # Linear basis function on the right node of the subinterval.
                phiR = (xx - xL) / (xR - xL)
                dphiR = 1.0 / (xR - xL)

            # Assemble local contributions to the matrices.
            # M => ∫ phi_i * phi_j
            # A => ∫ a(x) * dphi_i * dphi_j
            # B => ∫ p(x) * phi_i * phi_j
            # C => ∫ b(x) * phi_i * dphi_j
            if dofL is not None:
                M_loc[dofL, dofL] += w * (phiL * phiL)
                A_loc[dofL, dofL] += w * (a_val * dphiL * dphiL)
                B_loc[dofL, dofL] += w * (p_val * phiL * phiL)
                C_loc[dofL, dofL] += w * (b_val * phiL * dphiL)
            if (dofL is not None) and (dofR is not None):
                M_loc[dofL, dofR] += w * (phiL * phiR)
                M_loc[dofR, dofL] += w * (phiR * phiL)
                A_loc[dofL, dofR] += w * (a_val * dphiL * dphiR)
                A_loc[dofR, dofL] += w * (a_val * dphiR * dphiL)
                B_loc[dofL, dofR] += w * (p_val * phiL * phiR)
                B_loc[dofR, dofL] += w * (p_val * phiR * phiL)
                C_loc[dofL, dofR] += w * (b_val * phiL * dphiR)
                C_loc[dofR, dofL] += w * (b_val * phiR * dphiL)
            if dofR is not None:
                M_loc[dofR, dofR] += w * (phiR * phiR)
                A_loc[dofR, dofR] += w * (a_val * dphiR * dphiR)
                B_loc[dofR, dofR] += w * (p_val * phiR * phiR)
                C_loc[dofR, dofR] += w * (b_val * phiR * dphiR)

    return M_loc, A_loc, B_loc, C_loc


# ---------------------------------------------------------------------------
# The following functions compute the "tripod" integrals at each vertex.
# For each vertex that touches an edge, we integrate over the subinterval
# corresponding to the "arm" of the vertex. Separate routines are provided for:
#   - The left side of an edge.
#   - The right side of an edge.
#   - Cross integrals that couple the vertex basis function with the edge basis function.
#   - Their corresponding convection terms.
# ---------------------------------------------------------------------------

def integrate_vertex_arm_left(h, a_func, p_func):
    """
    Integrate the contributions on the left arm of a vertex from an edge subinterval [0, h].
    
    Computes the local integrals for the mass, stiffness, and potential matrices.
    
    Parameters:
    -----------
    h : float
        Length of the arm (typically h = 1/(nx+1)).
    a_func : callable
        Coefficient function a(x) for the diffusion term.
    p_func : callable
        Potential function p(x) for the reaction term.
    
    Returns:
    --------
    tuple of floats
        (M_d, A_d, B_d): Integrated contributions for mass, stiffness, and potential.
    """
    gauss_pts = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
    gauss_wts = np.array([1.0, 1.0])
    M_d = 0; A_d = 0; B_d = 0
    mid = 0.5 * h
    half = 0.5 * h
    for (gp, gw) in zip(gauss_pts, gauss_wts):
        xx = mid + half * gp
        w = half * gw
        phi_v = 1.0 - xx / h      # Linear basis function for the vertex.
        dphi_v = -1.0 / h         # Derivative of the vertex basis function.
        a_val = a_func(xx)
        p_val = p_func(xx)
        M_d += w * (phi_v * phi_v)
        A_d += w * (a_val * dphi_v * dphi_v)
        B_d += w * (p_val * phi_v * phi_v)
    return M_d, A_d, B_d

def integrate_vertex_arm_left_convection(h, b_func):
    """
    Integrate the convection term on the left arm of a vertex from an edge subinterval [0, h].
    
    Parameters:
    -----------
    h : float
        Length of the arm.
    b_func : callable
        Convection coefficient function b(x).
    
    Returns:
    --------
    float
        Integrated convection contribution (diagonal term).
    """
    gauss_pts = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
    gauss_wts = np.array([1.0, 1.0])
    mid = 0.5 * h
    half = 0.5 * h
    cdiag = 0
    for gp, gw in zip(gauss_pts, gauss_wts):
        xx = mid + half * gp
        w = half * gw
        phi_v = 1.0 - xx / h
        dphi_v = -1.0 / h
        b_val = b_func(xx)
        cdiag += w * (b_val * phi_v * dphi_v)
    return cdiag

def integrate_vertex_arm_left_cross(h, a_func, p_func):
    """
    Compute the cross integral on the left arm, coupling the vertex and edge basis functions.
    
    This routine integrates over the interval [0, h] and computes contributions to the off-diagonal
    entries between vertex and edge dofs.
    
    Parameters:
    -----------
    h : float
        Length of the arm.
    a_func : callable
        Coefficient function for the diffusion term.
    p_func : callable
        Potential function for the reaction term.
    
    Returns:
    --------
    tuple of floats
        (m_c, a_c, b_c): Cross contributions to the mass, stiffness, and potential matrices.
    """
    gauss_pts = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
    gauss_wts = np.array([1.0, 1.0])
    mid = 0.5 * h
    half = 0.5 * h
    m_c = 0; a_c = 0; b_c = 0
    for gp, gw in zip(gauss_pts, gauss_wts):
        xx = mid + half * gp
        w = half * gw
        phi_v = 1.0 - xx / h       # Vertex basis function.
        dphi_v = -1.0 / h          # Its derivative.
        phi_e = xx / h             # Edge basis function on the arm.
        dphi_e = 1.0 / h           # Derivative of edge basis function.
        a_val = a_func(xx)
        p_val = p_func(xx)
        m_c += w * (phi_v * phi_e)
        a_c += w * (a_val * dphi_v * dphi_e)
        b_c += w * (p_val * phi_v * phi_e)
    return m_c, a_c, b_c

def integrate_vertex_arm_left_cross_convection(h, b_func):
    """
    Compute the convection cross term on the left arm, coupling vertex and edge contributions.
    
    Parameters:
    -----------
    h : float
        Length of the arm.
    b_func : callable
        Convection coefficient function.
    
    Returns:
    --------
    float
        Cross contribution for the convection term.
    """
    gauss_pts = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
    gauss_wts = np.array([1.0, 1.0])
    mid = 0.5 * h
    half = 0.5 * h
    c_cr = 0
    for gp, gw in zip(gauss_pts, gauss_wts):
        xx = mid + half * gp
        w = half * gw
        phi_v = 1.0 - xx / h
        dphi_e = 1.0 / h
        b_val = b_func(xx)
        c_cr += w * (b_val * phi_v * dphi_e)
    return c_cr

def integrate_vertex_arm_right(h, a_func, p_func):
    """
    Integrate the contributions on the right arm of a vertex from an edge subinterval [1-h, 1].
    
    Computes the local integrals for the mass, stiffness, and potential matrices on the right side.
    
    Parameters:
    -----------
    h : float
        Length of the arm.
    a_func : callable
        Diffusion coefficient function.
    p_func : callable
        Potential function.
    
    Returns:
    --------
    tuple of floats
        (M_d, A_d, B_d): Integrated contributions for the mass, stiffness, and potential matrices.
    """
    gauss_pts = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
    gauss_wts = np.array([1.0, 1.0])
    xL = 1.0 - h
    xR = 1.0
    mid = 0.5 * (xL + xR)
    half = 0.5 * (xR - xL)
    M_d = 0; A_d = 0; B_d = 0
    for gp, gw in zip(gauss_pts, gauss_wts):
        xx = mid + half * gp
        w = half * gw
        phi_v = (xx - xL) / h      # Right arm vertex basis function.
        dphi_v = 1.0 / h           # Its derivative.
        a_val = a_func(xx)
        p_val = p_func(xx)
        M_d += w * (phi_v * phi_v)
        A_d += w * (a_val * dphi_v * dphi_v)
        B_d += w * (p_val * phi_v * phi_v)
    return M_d, A_d, B_d

def integrate_vertex_arm_right_convection(h, b_func):
    """
    Integrate the convection term on the right arm of a vertex from an edge subinterval [1-h, 1].
    
    Parameters:
    -----------
    h : float
        Length of the arm.
    b_func : callable
        Convection coefficient function.
    
    Returns:
    --------
    float
        Integrated convection contribution.
    """
    gauss_pts = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
    gauss_wts = np.array([1.0, 1.0])
    xL = 1.0 - h
    xR = 1.0
    mid = 0.5 * (xL + xR)
    half = 0.5 * (xR - xL)
    c_val = 0
    for gp, gw in zip(gauss_pts, gauss_wts):
        xx = mid + half * gp
        w = half * gw
        phi_v = (xx - xL) / h
        dphi_v = 1.0 / h
        b_val = b_func(xx)
        c_val += w * (b_val * phi_v * dphi_v)
    return c_val

def integrate_vertex_arm_right_cross(h, a_func, p_func):
    """
    Compute the cross integral on the right arm, coupling the vertex and edge basis functions.
    
    Integrates over the interval [1-h, 1] to obtain contributions to off-diagonal entries between
    vertex and edge degrees of freedom.
    
    Parameters:
    -----------
    h : float
        Length of the arm.
    a_func : callable
        Diffusion coefficient function.
    p_func : callable
        Potential function.
    
    Returns:
    --------
    tuple of floats
        (m_c, a_c, b_c): Cross contributions to the mass, stiffness, and potential matrices.
    """
    gauss_pts = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
    gauss_wts = np.array([1.0, 1.0])
    xL = 1.0 - h
    xR = 1.0
    mid = 0.5 * (xL + xR)
    half = 0.5 * (xR - xL)
    m_c = 0; a_c = 0; b_c = 0
    for gp, gw in zip(gauss_pts, gauss_wts):
        xx = mid + half * gp
        w = half * gw
        phi_v = (xx - xL) / h      # Vertex basis function.
        dphi_v = 1.0 / h           # Its derivative.
        phi_e = 1.0 - phi_v        # Complementary edge basis function.
        dphi_e = -dphi_v          # Derivative of the edge basis function.
        a_val = a_func(xx)
        p_val = p_func(xx)
        m_c += w * (phi_v * phi_e)
        a_c += w * (a_val * dphi_v * dphi_e)
        b_c += w * (p_val * phi_v * phi_e)
    return m_c, a_c, b_c

def integrate_vertex_arm_right_cross_convection(h, b_func):
    """
    Compute the convection cross term on the right arm, coupling vertex and edge contributions.
    
    Parameters:
    -----------
    h : float
        Length of the arm.
    b_func : callable
        Convection coefficient function.
    
    Returns:
    --------
    float
        Cross contribution for the convection term.
    """
    gauss_pts = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
    gauss_wts = np.array([1.0, 1.0])
    xL = 1.0 - h
    xR = 1.0
    mid = 0.5 * (xL + xR)
    half = 0.5 * (xR - xL)
    c_cr = 0
    for gp, gw in zip(gauss_pts, gauss_wts):
        xx = mid + half * gp
        w = half * gw
        phi_v = (xx - xL) / h
        phi_e = 1.0 - phi_v
        dphi_e = -1.0 / h
        b_val = b_func(xx)
        c_cr += w * (b_val * phi_v * dphi_e)
    return c_cr

def buildFEM_matrices(edges, nx, all_vertices, a_coefficient, b_coefficient, p_potential):
    """
    Assemble the global finite element matrices for the PDE on a graph.
    
    The global system has degrees of freedom corresponding to:
      - Edge dofs: For each of the E edges, there are nx interior degrees of freedom.
      - Vertex dofs: Each vertex (both interior and boundary) gets an additional dof.
    
    In addition to the standard local edge contributions computed by build_local_edge_matrices_all,
    "tripod" integrals are computed on sub-intervals at the vertices (arms) and added to the global matrices.
    Cross integrals are also computed to couple vertex dofs with the boundary dofs on adjacent edges.
    
    Parameters:
    -----------
    edges : list of tuples
        List of edge connectivity tuples (s, t) extracted from the graph.
    nx : int
        Number of local degrees of freedom per edge.
    all_vertices : list
        Sorted list of all vertices in the graph.
    a_coefficient : callable
        Coefficient function a(x) for the diffusion term.
    b_coefficient : callable
        Coefficient function b(x) for the convection term.
    p_potential : callable
        Potential function p(x).
    
    Returns:
    --------
    tuple of np.ndarray
        Global matrices (M_H, A_H, B_H, C_H) of size (total_dof, total_dof), where
        total_dof = E * nx + number_of_vertices.
    """
    E = len(edges)
    total_dof = E * nx + len(all_vertices)

    # Build an adjacency list mapping each vertex to the indices of the edges it touches.
    max_v = max(all_vertices)
    adjacency_list = [[] for _ in range(max_v + 1)]
    for e_idx, (s, t) in enumerate(edges):
        adjacency_list[s].append(e_idx)
        adjacency_list[t].append(e_idx)

    # Initialize global matrices.
    M_H = np.zeros((total_dof, total_dof))
    A_H = np.zeros((total_dof, total_dof))
    B_H = np.zeros((total_dof, total_dof))
    C_H = np.zeros((total_dof, total_dof))

    # Assemble the edge contributions.
    dof_edge = []
    offset = 0
    for e_idx, (ss, tt) in enumerate(edges):
        # Compute local matrices for the current edge.
        Mloc, Aloc, Bloc, Cloc = build_local_edge_matrices_all(nx, a_coefficient, b_coefficient, p_potential)
        # Local degrees of freedom indices for this edge.
        rowcols = np.arange(offset, offset + nx)
        dof_edge.append(rowcols)
        offset += nx
        M_H[np.ix_(rowcols, rowcols)] += Mloc
        A_H[np.ix_(rowcols, rowcols)] += Aloc
        B_H[np.ix_(rowcols, rowcols)] += Bloc
        C_H[np.ix_(rowcols, rowcols)] += Cloc

    # Assign a unique degree of freedom for each vertex.
    dof_vertex = {}
    vstart = E * nx
    v2index = {v: i for (i, v) in enumerate(all_vertices)}
    for i, v in enumerate(all_vertices):
        dof_vertex[v] = vstart + i

    h = 1.0 / (nx + 1)  # Length of the subinterval used in vertex integrals.

    # Add "tripod" contributions for each vertex (applied to all vertices, including boundary).
    for v in all_vertices:
        v_dof = dof_vertex[v]
        M_diag = 0; A_diag = 0; B_diag = 0; C_diag = 0
        # Sum the contributions from each edge touching vertex v.
        for e_idx in adjacency_list[v]:
            (s, t) = edges[e_idx]
            if s == v:
                m_d, a_d, b_d = integrate_vertex_arm_left(h, a_coefficient, p_potential)
                c_d = integrate_vertex_arm_left_convection(h, b_coefficient)
            elif t == v:
                m_d, a_d, b_d = integrate_vertex_arm_right(h, a_coefficient, p_potential)
                c_d = integrate_vertex_arm_right_convection(h, b_coefficient)
            else:
                m_d = 0; a_d = 0; b_d = 0; c_d = 0
            M_diag += m_d
            A_diag += a_d
            B_diag += b_d
            C_diag += c_d
        # Add the integrated vertex contributions to the diagonal of the global matrices.
        M_H[v_dof, v_dof] += M_diag
        A_H[v_dof, v_dof] += A_diag
        B_H[v_dof, v_dof] += B_diag
        C_H[v_dof, v_dof] += C_diag

        # Compute and add the cross integrals between vertex dof and edge boundary dofs.
        for e_idx in adjacency_list[v]:
            (s, t) = edges[e_idx]
            if s == v:
                # For left endpoint of the edge.
                edge_boundary_dof = 0
                m_c, a_c, b_c = integrate_vertex_arm_left_cross(h, a_coefficient, p_potential)
                c_c = integrate_vertex_arm_left_cross_convection(h, b_coefficient)
            elif t == v:
                # For right endpoint of the edge.
                edge_boundary_dof = nx - 1
                m_c, a_c, b_c = integrate_vertex_arm_right_cross(h, a_coefficient, p_potential)
                c_c = integrate_vertex_arm_right_cross_convection(h, b_coefficient)
            else:
                continue
            # Get the global degree of freedom for the edge boundary.
            rowcols = dof_edge[e_idx]
            e_dof = rowcols[edge_boundary_dof]

            # Add the symmetric cross contributions.
            M_H[v_dof, e_dof] += m_c
            M_H[e_dof, v_dof] += m_c

            A_H[v_dof, e_dof] += a_c
            A_H[e_dof, v_dof] += a_c

            B_H[v_dof, e_dof] += b_c
            B_H[e_dof, v_dof] += b_c

            C_H[v_dof, e_dof] += c_c
            C_H[e_dof, v_dof] += c_c

    return M_H, A_H, B_H, C_H

def apply_dirichlet_bc(A_sys, RHS, boundary_vertices, all_vertices, t_val, E, nx, g):
    """
    Apply Dirichlet boundary conditions to the global system.

    For each boundary vertex (with a corresponding global degree of freedom),
    this function modifies the system matrix A_sys and the right-hand side (RHS) vector
    to enforce the Dirichlet condition:
         u(v, t_val) = g(t_val, v)
    
    The degree of freedom corresponding to a vertex v is given by:
         v_dof = E * nx + index_in_all_vertices(v)
    
    Parameters:
    -----------
    A_sys : np.ndarray
        Global system matrix to be modified.
    RHS : np.ndarray
        Right-hand side vector of the linear system.
    boundary_vertices : list
        List of vertices where Dirichlet boundary conditions are prescribed.
    all_vertices : list
        Sorted list of all vertices.
    t_val : float
        Current time value (used in the boundary condition function).
    E : int
        Number of edges.
    nx : int
        Number of local degrees of freedom per edge.
    g : callable
        Dirichlet boundary condition function, g(t, v).
    
    Returns:
    --------
    tuple
        The modified system matrix and RHS vector (A_sys, RHS) after applying Dirichlet BCs.
    """
    for v in boundary_vertices:
        # Compute the global degree of freedom index for vertex v.
        v_dof = E * nx + all_vertices.index(v)
        # Zero out the row corresponding to the boundary condition.
        A_sys[v_dof, :] = 0.0
        # Set the diagonal entry to 1 (enforcing u(v) = g(t_val, v)).
        A_sys[v_dof, v_dof] = 1.0
        # Set the RHS value to the boundary data.
        RHS[v_dof] = g(t_val, v)
    return A_sys, RHS
