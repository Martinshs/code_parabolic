import numpy as np

def a_coefficient(x):
    return 0.5 + x*(x-1)

def b_coefficient(x):
    return 0.5* np.sin(np.pi*x)

def p_potential(x):
    return np.sin(np.pi*x)

# define polynomials y1..y11 ignoring e^-t
# Define las funciones polinómicas de grado 4
def y_poly(edge_id, x):
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

# Define la primera derivada de los polinomios
def dy_poly_dx(edge_id, x):
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

# Define la segunda derivada de los polinomios
def d2y_poly_dx2(edge_id, x):
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

def y_edge(edge_id, t, x):
    return y_poly(edge_id,x)* np.sin(2*np.pi *t)

def partial_t_y_edge(edge_id, t, x):
    return 2*np.pi*y_poly(edge_id,x)* np.cos(2*np.pi *t)

def partial_x_y_edge(edge_id, t, x):
    return dy_poly_dx(edge_id,x)* np.sin(2*np.pi *t)

def d2_y_edge(edge_id, t, x):
    return d2y_poly_dx2(edge_id,x)* np.sin(2*np.pi *t)

def partial_x_a_partial_x_y_edge(edge_id,t,x):
    aVal= a_coefficient(x)
    da_dx= 2*x -1
    return da_dx* partial_x_y_edge(edge_id,t,x) + aVal*d2_y_edge(edge_id,t,x)

def f_edge(edge_id, t, x):
    val_t= partial_t_y_edge(edge_id,t,x)
    val_diff= partial_x_a_partial_x_y_edge(edge_id,t,x)
    val_conv= b_coefficient(x)* partial_x_y_edge(edge_id,t,x)
    val_pot= p_potential(x)* y_edge(edge_id,t,x)
    return val_t - val_diff + val_conv + val_pot




def initial_condition_dirichlet_full(edges, all_vertices,nx):
    
    E = len(edges)
    n_vts =  len(all_vertices)
    total_dof = E*nx + n_vts
    y0= np.zeros(total_dof)
    # (A) edge interior dofs
    dx= 1.0/(nx+1)
    for i_edge in range(E):
        edge_offset= i_edge* nx
        edge_id= i_edge + 1
        x_nodes= np.linspace(dx,1.0- dx, nx)
        for k2 in range(nx):
            x_val= x_nodes[k2]
            y0[edge_offset + k2]= y_edge(edge_id, 0, x_val)

    # (B) vertex dofs
    # We'll define adjacency so we can see if vertex v is 's' => x=0 or 't' => x=1 in a particular edge
    # then call y_edge(edge_id, 0, 0) or y_edge(edge_id, 0, 1).
    # If multiple edges meet at v, we can pick the first or any, hoping they are consistent.
    max_v= max(all_vertices)
    adjacency_list= [[] for _ in range(max_v+1)]
    for e_idx,(s,t) in enumerate(edges):
        adjacency_list[s].append( (e_idx, 's') )  # 's' means this vertex is the start
        adjacency_list[t].append( (e_idx, 't') )

    # dof for each vertex => dof_vertex= E*nx + index in all_vertices
    def vertex_dof_index(v):
        return E*nx + all_vertices.index(v)

    for v in all_vertices:
        v_dof= vertex_dof_index(v)
        # Find an incident edge to figure out if v is start (x=0) or end (x=1)
        # We'll just pick adjacency_list[v][0] if it exists
        if adjacency_list[v]:
            (e_idx, pos) = adjacency_list[v][0]  # pick the first edge
            edge_id= e_idx+1
            if pos=='s':
                # x=0 => y_edge(edge_id,0,0)
                init_val= y_edge(edge_id, 0, 0.0)
            else:
                # x=1 => y_edge(edge_id,0,1)
                init_val= y_edge(edge_id, 0, 1.0)
        else:
            # no edges?? should not happen
            init_val= 0.0

        y0[v_dof]= init_val  # exact sol at t=0 for vertex v
    return y0

def compute_exact_solution(edges, all_vertices,interior_vertices, boundary_vertices, nx, t):
    """
    same dimension => E*nx + #all_vertices
    """

    E= len(edges)
    total_dof= E*nx+ len(all_vertices)
    y_ex= np.zeros(total_dof)

    dx= 1.0/(nx+1)
    for i_edge in range(E):
        offset= i_edge*nx
        edge_id= i_edge+1
        x_nodes= np.linspace(dx,1.0- dx, nx)
        for k2 in range(nx):
            x_val= x_nodes[k2]
            y_ex[offset+ k2]= y_edge(edge_id, t, x_val)

    # vertex dofs => dof= E*nx + index => if boundary => e^-t, else polynomial if you want or 0 
    # But let's define the polynomial approach if we want. 
    for v in all_vertices:
        for i, (iv,fv) in enumerate(edges):
            if v==iv:
                v_dof = E*nx + v
                y_ex[v_dof]= y_edge(i+1, t, 0)
            elif v==fv:
                v_dof = E*nx + v
                y_ex[v_dof]= y_edge(i+1, t, 1)

    return y_ex

# Function 1: Define node positions
def define_positions_example_1():
    positions = {
        'v1': [-1.9, 7.95],
        'v2': [-1.9, 2.05],
        'v3': [1.3 , 5],        
        'v4': [6   , 5],
        'v5': [9.5 , 7.95],
        'v6': [10  , 2.05],
        'v7': [14  , 5],
        'v8': [18.7, 5],
        'v9': [21.9, 7.95],  
        'v10':[21.9, 2.05]
    }
    return positions

def g_boundary_condition_ex1(t,v):
    return np.sin(2*np.pi *t)