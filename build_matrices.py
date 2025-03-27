import numpy as np

#############################################################################
# 3) Build M,A,B,C for edges + "tripod" for ALL vertices
#    => both boundary and interior vertices get dofs
##############################################################################
# We'll replicate your "build_local_edge_matrices_all" for edges,
# then do "tripod" integrals for each vertex that touches an edge endpoint.

def build_local_edge_matrices_all(nx, a_func, b_func, p_func):
    M_loc= np.zeros((nx,nx))
    A_loc= np.zeros((nx,nx))
    B_loc= np.zeros((nx,nx))
    C_loc= np.zeros((nx,nx))

    dx= 1.0/(nx+1)
    gauss_pts= np.array([-1/np.sqrt(3),1/np.sqrt(3)])
    gauss_wts= np.array([1.0,1.0])

    def interior_index(k):
        if 1<=k<=nx:
            return k-1
        return None

    for i_sub in range(nx+1):
        xL= i_sub*dx
        xR= (i_sub+1)* dx
        dofL= interior_index(i_sub)
        dofR= interior_index(i_sub+1)
        if (dofL is None) and (dofR is None):
            continue
        mid= 0.5*(xL+ xR)
        half=0.5*(xR- xL)
        for (gp,gw) in zip(gauss_pts,gauss_wts):
            xx= mid+ half*gp
            w= half*gw
            a_val= a_func(xx)
            b_val= b_func(xx)
            p_val= p_func(xx)

            phiL=0; dphiL=0
            if dofL is not None:
                phiL= (xR-xx)/(xR- xL)
                dphiL= -1.0/(xR- xL)
            phiR=0; dphiR=0
            if dofR is not None:
                phiR= (xx- xL)/(xR- xL)
                dphiR= +1.0/(xR- xL)

            # M => \int phi_i phi_j
            # A => \int a_val dphi_i dphi_j
            # B => \int p_val phi_i phi_j
            # C => \int b_val phi_i dphi_j
            if dofL is not None:
                M_loc[dofL,dofL]+= w*( phiL* phiL )
                A_loc[dofL,dofL]+= w*( a_val*dphiL*dphiL )
                B_loc[dofL,dofL]+= w*( p_val* phiL* phiL )
                C_loc[dofL,dofL]+= w*( b_val* phiL*dphiL )
            if (dofL is not None) and (dofR is not None):
                M_loc[dofL,dofR]+= w*( phiL* phiR )
                M_loc[dofR,dofL]+= w*( phiR* phiL )

                A_loc[dofL,dofR]+= w*( a_val*dphiL*dphiR )
                A_loc[dofR,dofL]+= w*( a_val*dphiR*dphiL )

                B_loc[dofL,dofR]+= w*( p_val* phiL*phiR )
                B_loc[dofR,dofL]+= w*( p_val* phiR*phiL )

                C_loc[dofL,dofR]+= w*( b_val* phiL*dphiR )
                C_loc[dofR,dofL]+= w*( b_val* phiR*dphiL )
            if dofR is not None:
                M_loc[dofR,dofR]+= w*( phiR* phiR )
                A_loc[dofR,dofR]+= w*( a_val*dphiR*dphiR )
                B_loc[dofR,dofR]+= w*( p_val* phiR* phiR )
                C_loc[dofR,dofR]+= w*( b_val* phiR*dphiR )

    return M_loc,A_loc,B_loc,C_loc

# We'll define integrals for each vertex dof => including boundary vertices now.
# The difference from prior code: we *do not skip* boundary in "tripod." 
# We'll do the same approach for each vertex that touches an edge => sub-interval [0,h] or [1-h,1].

# We'll define the same "integrate_vertex_arm_*" routines as before, then combine them in buildFEM_matrices.


def integrate_vertex_arm_left(h, a_func, p_func):
    gauss_pts= np.array([-1/np.sqrt(3),1/np.sqrt(3)])
    gauss_wts= np.array([1.0,1.0])
    M_d=0; A_d=0; B_d=0
    mid=0.5*h
    half=0.5*h
    for (gp,gw) in zip(gauss_pts,gauss_wts):
        xx= mid+ half*gp
        w= half*gw
        phi_v= 1.0- xx/h
        dphi_v= -1.0/h
        a_val= a_func(xx)
        p_val= p_func(xx)
        M_d+= w*( phi_v* phi_v )
        A_d+= w*( a_val* dphi_v*dphi_v )
        B_d+= w*( p_val* phi_v* phi_v )
    return M_d,A_d,B_d

def integrate_vertex_arm_left_convection(h, b_func):
    gauss_pts= np.array([-1/np.sqrt(3),1/np.sqrt(3)])
    gauss_wts= np.array([1.0,1.0])
    mid=0.5*h
    half=0.5*h
    cdiag=0
    for gp,gw in zip(gauss_pts,gauss_wts):
        xx= mid+ half*gp
        w= half*gw
        phi_v= 1.0- xx/h
        dphi_v= -1.0/h
        b_val= b_func(xx)
        cdiag+= w*( b_val* phi_v*dphi_v )
    return cdiag

def integrate_vertex_arm_left_cross(h, a_func, p_func):
    gauss_pts= np.array([-1/np.sqrt(3),1/np.sqrt(3)])
    gauss_wts= np.array([1.0,1.0])
    mid=0.5*h
    half=0.5*h
    m_c=0;a_c=0;b_c=0
    for gp,gw in zip(gauss_pts,gauss_wts):
        xx= mid+ half*gp
        w= half*gw
        phi_v= 1.0- xx/h
        dphi_v= -1.0/h
        phi_e= xx/h
        dphi_e= 1.0/h
        a_val= a_func(xx)
        p_val= p_func(xx)
        m_c+= w*( phi_v* phi_e )
        a_c+= w*( a_val* dphi_v*dphi_e )
        b_c+= w*( p_val* phi_v* phi_e )
    return m_c,a_c,b_c

def integrate_vertex_arm_left_cross_convection(h, b_func):
    gauss_pts= np.array([-1/np.sqrt(3),1/np.sqrt(3)])
    gauss_wts= np.array([1.0,1.0])
    mid=0.5*h
    half=0.5*h
    c_cr=0
    for gp,gw in zip(gauss_pts,gauss_wts):
        xx= mid+ half*gp
        w= half*gw
        phi_v= 1.0- xx/h
        dphi_e= 1.0/h
        b_val= b_func(xx)
        c_cr+= w*( b_val* phi_v*dphi_e )
    return c_cr

def integrate_vertex_arm_right(h, a_func, p_func):
    gauss_pts= np.array([-1/np.sqrt(3),1/np.sqrt(3)])
    gauss_wts= np.array([1.0,1.0])
    xL=1.0- h
    xR=1.0
    mid=0.5*(xL+xR)
    half=0.5*(xR- xL)
    M_d=0; A_d=0; B_d=0
    for gp,gw in zip(gauss_pts,gauss_wts):
        xx= mid+ half*gp
        w= half*gw
        phi_v= (xx- xL)/h
        dphi_v= 1.0/h
        a_val= a_func(xx)
        p_val= p_func(xx)
        M_d+= w*( phi_v* phi_v )
        A_d+= w*( a_val*dphi_v*dphi_v )
        B_d+= w*( p_val* phi_v* phi_v )
    return M_d,A_d,B_d

def integrate_vertex_arm_right_convection(h, b_func):
    gauss_pts= np.array([-1/np.sqrt(3),1/np.sqrt(3)])
    gauss_wts= np.array([1.0,1.0])
    xL=1.0- h
    xR=1.0
    mid=0.5*(xL+xR)
    half=0.5*(xR- xL)
    c_val=0
    for gp,gw in zip(gauss_pts,gauss_wts):
        xx= mid+ half*gp
        w= half*gw
        phi_v= (xx- xL)/h
        dphi_v= 1.0/h
        b_val= b_func(xx)
        c_val+= w*( b_val* phi_v*dphi_v )
    return c_val

def integrate_vertex_arm_right_cross(h, a_func, p_func):
    gauss_pts= np.array([-1/np.sqrt(3),1/np.sqrt(3)])
    gauss_wts= np.array([1.0,1.0])
    xL=1.0- h
    xR=1.0
    mid=0.5*(xL+xR)
    half=0.5*(xR- xL)
    m_c=0;a_c=0;b_c=0
    for gp,gw in zip(gauss_pts,gauss_wts):
        xx= mid+ half*gp
        w= half*gw
        phi_v= (xx- xL)/h
        dphi_v= 1.0/h
        phi_e= 1.0- phi_v
        dphi_e= -dphi_v
        a_val= a_func(xx)
        p_val= p_func(xx)
        m_c+= w*( phi_v* phi_e )
        a_c+= w*( a_val*dphi_v*dphi_e )
        b_c+= w*( p_val* phi_v* phi_e )
    return m_c,a_c,b_c

def integrate_vertex_arm_right_cross_convection(h, b_func):
    gauss_pts= np.array([-1/np.sqrt(3),1/np.sqrt(3)])
    gauss_wts= np.array([1.0,1.0])
    xL=1.0- h
    xR=1.0
    mid=0.5*(xL+xR)
    half=0.5*(xR- xL)
    c_cr=0
    for gp,gw in zip(gauss_pts,gauss_wts):
        xx= mid+ half*gp
        w= half*gw
        phi_v= (xx- xL)/h
        phi_e= 1.0- phi_v
        dphi_e= -1.0/h
        b_val= b_func(xx)
        c_cr+= w*( b_val* phi_v*dphi_e )
    return c_cr

def buildFEM_matrices(edges, nx, all_vertices, a_coefficient, b_coefficient, p_potential):
    """
    We do dofs => E*nx + #all_vertices. Each vertex => a dof. Then we add "tripod" integrals for each vertex-edge boundary sub-interval, no matter if interior or boundary.
    """
    E= len(edges)
    total_dof= E*nx + len(all_vertices)

    # adjacency
    max_v= max(all_vertices)
    adjacency_list= [[] for _ in range(max_v+1)]
    for e_idx,(s,t) in enumerate(edges):
        adjacency_list[s].append(e_idx)
        adjacency_list[t].append(e_idx)

    M_H= np.zeros((total_dof,total_dof))
    A_H= np.zeros((total_dof,total_dof))
    B_H= np.zeros((total_dof,total_dof))
    C_H= np.zeros((total_dof,total_dof))

    # edge dofs
    dof_edge=[]
    offset=0
    for e_idx,(ss,tt) in enumerate(edges):
        Mloc,Aloc,Bloc,Cloc= build_local_edge_matrices_all(nx,a_coefficient,b_coefficient,p_potential)
        rowcols= np.arange(offset, offset+nx)
        dof_edge.append(rowcols)
        offset+= nx
        M_H[np.ix_(rowcols,rowcols)] += Mloc
        A_H[np.ix_(rowcols,rowcols)] += Aloc
        B_H[np.ix_(rowcols,rowcols)] += Bloc
        C_H[np.ix_(rowcols,rowcols)] += Cloc

    # vertex dofs => each vertex => dof_index= E*nx + (index in all_vertices)
    dof_vertex={}
    vstart= E*nx
    v2index= { v: i for (i,v) in enumerate(all_vertices) }

    for (i,v) in enumerate(all_vertices):
        dof_vertex[v]= vstart + i

    h= 1.0/(nx+1)
    # "tripod" for each vertex, boundary or interior
    for v in all_vertices:
        v_dof= dof_vertex[v]
        M_diag=0; A_diag=0; B_diag=0; C_diag=0
        for e_idx in adjacency_list[v]:
            (s,t)= edges[e_idx]
            if s==v:
                m_d,a_d,b_d= integrate_vertex_arm_left(h, a_coefficient, p_potential)
                c_d= integrate_vertex_arm_left_convection(h, b_coefficient)
            elif t==v:
                m_d,a_d,b_d= integrate_vertex_arm_right(h, a_coefficient, p_potential)
                c_d= integrate_vertex_arm_right_convection(h, b_coefficient)
            else:
                m_d=0;a_d=0;b_d=0;c_d=0
            M_diag+= m_d
            A_diag+= a_d
            B_diag+= b_d
            C_diag+= c_d
        M_H[v_dof,v_dof]+= M_diag
        A_H[v_dof,v_dof]+= A_diag
        B_H[v_dof,v_dof]+= B_diag
        C_H[v_dof,v_dof]+= C_diag

        # cross integrals
        for e_idx in adjacency_list[v]:
            (s,t)= edges[e_idx]
            if s==v:
                edge_boundary_dof= 0
                m_c,a_c,b_c= integrate_vertex_arm_left_cross(h, a_coefficient, p_potential)
                c_c= integrate_vertex_arm_left_cross_convection(h, b_coefficient)
            elif t==v:
                edge_boundary_dof= nx-1
                m_c,a_c,b_c= integrate_vertex_arm_right_cross(h, a_coefficient, p_potential)
                c_c= integrate_vertex_arm_right_cross_convection(h, b_coefficient)
            else:
                continue
            rowcols= dof_edge[e_idx]
            e_dof= rowcols[edge_boundary_dof]

            M_H[v_dof,e_dof]+= m_c
            M_H[e_dof,v_dof]+= m_c

            A_H[v_dof,e_dof]+= a_c
            A_H[e_dof,v_dof]+= a_c

            B_H[v_dof,e_dof]+= b_c
            B_H[e_dof,v_dof]+= b_c

            C_H[v_dof,e_dof]+= c_c
            C_H[e_dof,v_dof]+= c_c

    return M_H, A_H, B_H, C_H


def apply_dirichlet_bc(A_sys, RHS, boundary_vertices, all_vertices, t_val, E, nx, g):
    """
    We have a dof for each vertex => dof_vertex= E*nx + indexInAllVertices
    We set row => [0..0,1,0..0], RHS= e^-t
    """
    for v in boundary_vertices:
        # dof index
        v_dof= E*nx + all_vertices.index(v)
        # zero out row
        A_sys[v_dof,:]= 0.0
        A_sys[v_dof,v_dof]= 1.0
     
        RHS[v_dof]= g(t_val, v)
    return A_sys, RHS