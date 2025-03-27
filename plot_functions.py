from data_ex1 import compute_exact_solution
from utils import check_folder
from graph import define_graph_full

import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_edges(computed_sol,nx, nt, T, full_edges, show=True, save =False, example = 'example_0'):  
    t_index = int(nt/3)+1
    mesht = np.linspace(0,T,nt)
    edges, interior_vertices, boundary_vertices= define_graph_full(full_edges)
    all_vertices = interior_vertices + boundary_vertices
    all_vertices.sort()
    y_ex= compute_exact_solution(edges, all_vertices,interior_vertices, boundary_vertices, nx, t=mesht[t_index])

    # Mosaic
    fig, axs= plt.subplot_mosaic(
                    [['y1', 'y2','y3', 'y4', 'y5'],
                    ['y6', 'y7','y8', 'y9', 'y10']],
                        dpi=100,figsize=(18, 5), gridspec_kw={
                        "top": 0.9,
                        "left": 0.2,
                        "right": 0.8,
                        "wspace": 0.17,
                        "hspace": 0.45,},)
    fig.suptitle(f'Solution on each edge at t={mesht[t_index]:.02f}', x=0.5, y=1.1,size = 15)

    #axs= axs.ravel()
    dx= 1.0/(nx+1)
    x_nodes= np.linspace(dx, 1.0- dx, nx)
    for e_idx,(s,t2) in enumerate(full_edges.values()):
        offset= e_idx*nx
        offset_end= offset+ nx
        na_e_idx = 'y'+str(e_idx+1) 
        ax= axs[na_e_idx]
        ax.plot(x_nodes, y_ex[offset: offset_end], 'k', label='Exact solution', linewidth=4)
        ax.plot(x_nodes, computed_sol[t_index, offset: offset_end], 'r--', label='Computed solution', linewidth=3)
        ax.set_title(f"Edge {e_idx+1}: (v{s+1}->v{t2+1})")
        ax.grid(True)
        ax.set_facecolor('#f5f5f5')  # Light grey background
        if e_idx==2:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5),
                ncol=2, fancybox=True, shadow=False, prop={'size': 13}) 
        ax.set_ylim(-1,2.3)
    if save:
        newpath = check_folder(example)
        name = os.path.join(newpath,f'plot_each_edge_'+str(t_index)+'.png')
        plt.savefig(name, dpi=150, bbox_inches='tight' )   
    if show: 
        plt.show()
    else:
        plt.close()
    return None




# Function 2: Set up edges, labels, and vertices
def setup_graph_from_full_edges(full_edges):
    edges = []
    edge_labels = {}
    # Sort keys by the numerical part to maintain order (optional)
    for key in sorted(full_edges.keys(), key=lambda k: int(k[1:])):
        i, j = full_edges[key]
        start, end = f'v{i+1}', f'v{j+1}'
        edge = (start, end)
        edges.append(edge)
        # Construct the label in LaTeX format
        edge_labels[edge] = f'$e_{{{int(key[1:])}}}$'
    return edges, edge_labels


# Function to compute Euclidean distance between two points
def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p2) - np.array(p1))

# Function to parameterize the edge and plot the curve with a specified color
def plot_curve_on_edge(ax, pos_start, pos_end,z_curve , color, num_points=100, linestyle = 'solid', linewidth = 3, zorder=0):
    # Compute Euclidean distance (length of the edge)
    a = euclidean_distance(pos_start, pos_end)
    
    # Parameterize the edge: we use t in [0, a]
    t = np.linspace(0, a, num_points)
    
    # Unit vector in the direction of the edge
    direction = np.array(pos_end) - np.array(pos_start)
    direction = direction / np.linalg.norm(direction)  # Normalize
    
    # Parameterize the line in R^2 along t (for the x and y directions)
    x_line = pos_start[0] + direction[0] * t
    y_line = pos_start[1] + direction[1] * t
    
    # Evaluate the function f along the interval [0, a]
    #z_curve = f(t, a)  # Pass 'a' (length of the edge) to the function
    
    # Plot the curve with the given color above the edge in the z direction
    ax.plot(x_line, y_line, z_curve, color=color, linewidth=linewidth, linestyle = linestyle, zorder=zorder)




# Plot function
def plot_graph_3d_with_curves(computed_sol, full_edges, nx, nt, T, k, positions, show=True, elevation=30, azimuth=30, zorder=0, save=False, example="example_0_3d"):
    
    edges, edge_labels  = setup_graph_from_full_edges(full_edges)

    node_radius = 0.15
    fig = plt.figure(figsize=(12, 6), dpi=100)
    ax = fig.add_subplot(projection='3d')

    # Plot graph
    for start, end in edges:
        pos_start = np.array(positions[start] + [0])
        pos_end = np.array(positions[end] + [0])
        
        vector = pos_end - pos_start
        direction = vector / np.linalg.norm(vector)
        arrow_start = pos_start + direction * node_radius
        arrow_end = pos_end - direction * node_radius
        
        ax.quiver(arrow_start[0], arrow_start[1], arrow_start[2],
                  arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1], arrow_end[2] - arrow_start[2],
                  color='k', linewidth=3, arrow_length_ratio=0, alpha=0.8)
    
    bigg_nodes = ['v1','v2','v3','v5','v6','v8','v9','v10']
    for node, pos in positions.items():
        x, y = pos
        z = 0
        if node in bigg_nodes:
            ax.scatter(x, y, z, s=50, facecolors='k', edgecolors='black', linewidths=1, alpha=0.4)
        else:    
            ax.scatter(x, y, z, s=20, facecolors='k', edgecolors='black', linewidths=1, alpha=0.4)
            
            
    # Add labels for each edge
    for (start, end) in edges:
        pos_start = np.array(positions[start])
        pos_end = np.array(positions[end])
        midpoint = (pos_start + pos_end) / 2.0
        dx, dy = pos_end - pos_start
        norm = np.hypot(dx, dy)
        if norm == 0:
            continue  # avoid division by zero
        perp = np.array([-dy, dx]) / norm
        epsilon = -0.7  # adjust offset as needed
        label_pos = midpoint + epsilon * perp
        # Place the label at z=0 to ensure readability
        label_text = edge_labels.get((start, end), '')
        ax.text(label_pos[0], label_pos[1], 0, label_text, fontsize=12, ha='center', va='center', color='black')

    # Compute exact solution
    edges_2, interior_vertices, boundary_vertices= define_graph_full(full_edges)
    all_vertices = interior_vertices + boundary_vertices
    mesht = np.linspace(0,T,nt)


    y_ex= compute_exact_solution(edges_2, all_vertices, interior_vertices, 
                                 boundary_vertices, nx,t=mesht[k])
    
    # Exact solution plot
    for i, (s, t) in enumerate(full_edges.values()):
        e_loc = (f'v{s+1}', f'v{t+1}')
        if e_loc in edges:
            y_loc_ex = y_ex[i*nx: i*nx+nx]
            plot_curve_on_edge(ax, positions[e_loc[0]], positions[e_loc[1]], y_loc_ex, '#153863', num_points=nx, zorder=zorder)

    # Computed solution plot
    for i, (s, t) in enumerate(full_edges.values()):
        e_loc = (f'v{s+1}', f'v{t+1}')
        if e_loc in edges:
            computed_sol_loc = computed_sol[k][i*nx: i*nx+nx]
            plot_curve_on_edge(ax, positions[e_loc[0]], positions[e_loc[1]],
                               computed_sol_loc, 'C3', num_points=nx, linestyle='dashed', linewidth=2, zorder=zorder)

    # Plot vertical lines at the vertices
    for v in all_vertices:
        v_name = f'v{v+1}' 
        if v_name in positions:
            E = len(full_edges)
            y_ex_vertex = y_ex[E*nx+v]
            x_point = [positions[v_name][0], positions[v_name][0]]
            y_point = [positions[v_name][1], positions[v_name][1]]
            ax.plot(x_point, y_point, [0, y_ex_vertex], color='k', linewidth=1, linestyle='dashed')

    # Ajuste limits
    ax.set_zlim(-2, 2)
    ax.set_ylim(1, 10)
    ax.set_xlim(-2, 24.5)
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1.5, 1, 0.5, 1]))

    # Rotation and elevation
    ax.view_init(elev=elevation, azim=azimuth)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # --- Legend ---
    from matplotlib.lines import Line2D
    legend_handles = [Line2D([0], [0], color='#153863', lw=3, linestyle='solid', label='Exact solution'),
                      Line2D([0], [0], color='C3', lw=3, linestyle='dashed', label='Computed solution')]
    ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.6, 0.89), ncol=2, fontsize=12)

    # --- Ajuste title ---
    fig.suptitle(F'Real solution vs Computed solution at time t={mesht[k]:.02f}', x =0.55, y=0.89, fontsize=16)
    plt.subplots_adjust(top=0.92)
    
    if save:
        newpath = check_folder(example)
        N_dig = len(str(mesht.shape[0]))
        name = os.path.join(newpath,f'plot_superposition'+str(k).zfill(N_dig)+'.png')
        plt.savefig(name, dpi=150, bbox_inches='tight' )

    if show:
        plt.show()
    else:
        plt.close()
    return None
    