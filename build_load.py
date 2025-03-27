import numpy as np

##############################################################################
# 4) PDE load vector => edges+ vertex dofs
##############################################################################


def compute_load_vector(f_edge, edges, nx, all_vertices, t):
    """
    Constructs the global load vector F for the PDE source term f_edge(edge_id, t, x)
    on a 1D graph with:
      - 'nx' interior nodes per edge,
      - a dof for each interior node on each edge,
      - a dof for each vertex in 'all_vertices'.

    Returns an array F of size [E*nx + len(all_vertices)], 
    where E=number_of_edges, each edge has 'nx' interior dofs, 
    and there's one dof per vertex.
    """ 
    # E = number of edges in the graph
    E = len(edges)

    # total_dof = (edge-based dofs) + (vertex-based dofs)
    total_dof = E*nx + len(all_vertices)

    # Initialize the load vector F with zeros
    F = np.zeros(total_dof)

    # We'll use 2-point Gauss integration on each sub-interval, 
    # so define the standard Gauss points and weights on [-1/sqrt(3), +1/sqrt(3)]
    gauss_pts = np.array([-1/np.sqrt(3), +1/np.sqrt(3)])
    gauss_wts = np.array([1.0, 1.0])

    # 'dx' is the uniform sub-interval length if the edge is parameterized [0,1] => we have (nx+1) sub-intervals
    dx = 1.0/(nx+1)

    ##########################################################################
    #  A) Contribute to the load for interior-edge dofs 
    #     (i.e. the standard "element by element" integration).
    ##########################################################################

    # First define a helper that maps sub-interval index k => dof index (or None if boundary).
    def interior_index(k):
        """
        If k=0 => boundary node, no interior dof.
        If 1 <= k <= nx => interior dof => index = k-1 
        If k = nx+1 => boundary node => no interior dof
        """
        if 1 <= k <= nx:
            return k - 1
        return None

    # Loop over each edge i_edge. The PDE has E edges => edges[i_edge] = (s, t2).
    for i_edge, (s, t2) in enumerate(edges):
        # offset for dofs in this edge: edge i_edge => dofs [offset.. offset+nx-1]
        offset = i_edge * nx
        # The PDE's source is associated with edge_id = i_edge+1 
        edge_id = i_edge + 1

        # We subdivide [0,1] of this edge into (nx+1) sub-intervals => i_sub=0.. nx
        for i_sub in range(nx+1):
            xL = i_sub * dx    # left boundary of sub-interval in [0,1]
            xR = (i_sub+1) * dx
            # Convert sub-interval index => local dof indices dofL, dofR
            dofL = interior_index(i_sub)
            dofR = interior_index(i_sub+1)

            # if dofL and dofR are both None => it's a boundary sub-interval => no "interior" dofs
            if (dofL is None) and (dofR is None):
                continue

            # We'll do 2-pt Gauss on this sub-interval => compute mid, half
            mid = 0.5*(xL + xR)
            half = 0.5*(xR - xL)

            # Now integrate f_edge(edge_id,t, x) * basis_i(x) over [xL, xR]
            for (gp, gw) in zip(gauss_pts, gauss_wts):
                # Map gp in [-1,1] => physical x in [xL, xR]
                xx = mid + half*gp
                w = half*gw   # scaled weight

                # Evaluate PDE source f at that x => f_edge depends on (edge_id, t, x)
                val_f = f_edge(edge_id, t, xx)

                # For an interior dof dofL => basis = (xR-xx)/(xR- xL)
                # For dofR => basis = (xx- xL)/(xR- xL)
                if dofL is not None:
                    phiL = (xR - xx)/(xR - xL)
                    F[offset + dofL] += w * ( val_f * phiL )

                if dofR is not None:
                    phiR = (xx - xL)/(xR - xL)
                    F[offset + dofR] += w * ( val_f * phiR )

    ##########################################################################
    #  B) Contribute to the load for vertex-based dofs 
    #     (the "tripod" shape function integrals).
    ##########################################################################

    # Next, we do the vertex dofs. Each vertex v has a "tripod" shape 
    # on boundary sub-interval [0,dx] or [1-dx,1] of each incident edge.

    # Build adjacency_list => for each vertex v, store the edges that meet v
    max_v = max(all_vertices)
    adjacency_list = [[] for _ in range(max_v+1)]
    for e_idx, (s, t2) in enumerate(edges):
        adjacency_list[s].append(e_idx)
        adjacency_list[t2].append(e_idx)

    # We define a dof map for each vertex => dof index = E*nx + index_in_all_vertices
    v2index = {v: i for (i, v) in enumerate(all_vertices)}
    def vertex_dof_index(v):
        """
        The dof index for vertex v is offset by E*nx (the sum of all interior-edge dofs),
        plus the index of v in the sorted list all_vertices.
        """
        return E*nx + v2index[v]

    # Now loop over each vertex => integrate the PDE source over the boundary sub-interval
    # of each incident edge's param [0,dx] or [1-dx,1], using the "tripod" shape basis function.
    for v in all_vertices:
        v_dof = vertex_dof_index(v)
        load_v = 0  # local accumulator for this vertex dof

        # For each edge e_idx that touches v, we see if v is s => param x=0 or t => param x=1
        for e_idx in adjacency_list[v]:
            (sv, tv) = edges[e_idx]
            edge_id = e_idx + 1

            if sv == v:
                # If v is 's', that means param x=0 => sub-interval [0,dx].
                # Then the shape function is phi_v(x) = 1 - x/dx in [0, dx].
                mid = 0.5* dx
                half= 0.5* dx
                # We'll do 2-point Gauss on that short sub-interval [0, dx].
                # We define the same gauss_pts, gauss_wts as for a unit interval, 
                # but we scale to length dx.
                gauss_pts= np.array([-1/np.sqrt(3), +1/np.sqrt(3)])
                gauss_wts= np.array([1.0,1.0])

                for (gp, gw) in zip(gauss_pts, gauss_wts):
                    xx= mid + half*gp   # map [-1,1] => [0, dx]
                    w= half*gw
                    # shape function at that local x => phi_v= 1 - xx/dx
                    phi_v= 1.0 - xx/dx
                    val_f= f_edge(edge_id, t, xx)  # PDE source
                    load_v += w*( val_f * phi_v )

            elif tv == v:
                # If v is 't', that means param x=1 => sub-interval [1-dx,1].
                # shape function => phi_v= (x - (1-dx))/dx in [1-dx,1].
                xL= 1.0 - dx
                xR= 1.0
                mid= 0.5*(xL+ xR)
                half=0.5*(xR- xL)

                gauss_pts= np.array([-1/np.sqrt(3), +1/np.sqrt(3)])
                gauss_wts= np.array([1.0,1.0])

                for (gp,gw) in zip(gauss_pts,gauss_wts):
                    xx= mid+ half*gp  # mapped to [1-dx,1]
                    w= half*gw
                    phi_v= (xx - xL)/ dx
                    val_f= f_edge(edge_id, t, xx)
                    load_v += w*( val_f * phi_v )

        # After summing over all incident edges, store in F for this vertex dof
        F[v_dof]+= load_v

    # Return the fully assembled load vector
    return F
