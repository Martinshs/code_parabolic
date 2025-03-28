from data_ex1 import compute_exact_solution
from utils import check_folder
from graph import define_graph_full

import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =============================================================================
# Function 1: Plot the solution on individual edges (2D plots)
# =============================================================================

def plot_edges(computed_sol, exact_sol, nx, nt, T, t_index, full_edges, show=True, save=False, example='example_0'):
    """
    Plot the computed and exact solutions on each edge of the graph at a given time.

    The function creates a mosaic of subplots (arranged in 2 rows and 5 columns) for the 10 edges.
    For each edge, it plots:
      - The exact solution computed via compute_exact_solution.
      - The computed solution at the specified time index.
    
    Parameters:
    -----------
    computed_sol : np.ndarray
        2D array (nt x total_dof) containing the computed solution at all time steps.
    nx : int
        Number of interior degrees of freedom per edge.
    nt : int
        Number of time steps.
    T : float
        Final time of the simulation.
    t_index : int
        Index in the time mesh at which to plot the solution.
    full_edges : dict
        Dictionary mapping edge identifiers to tuples of vertex indices.
    show : bool, optional
        Whether to display the plot interactively (default is True).
    save : bool, optional
        Whether to save the plot as an image file (default is False).
    example : str, optional
        Example identifier used to form the save directory name.
    
    Returns:
    --------
    None
    """
    # Create time mesh and compute the exact solution at time t_index.
    mesht = np.linspace(0, T, nt)
    edges, interior_vertices, boundary_vertices = define_graph_full(full_edges)
    all_vertices = interior_vertices + boundary_vertices
    all_vertices.sort()
    y_ex = exact_sol(edges, all_vertices, interior_vertices, boundary_vertices, nx, t=mesht[t_index])
    
    # Create a mosaic of subplots for 10 edges.
    fig, axs = plt.subplot_mosaic(
        [['y1', 'y2', 'y3', 'y4', 'y5'],
         ['y6', 'y7', 'y8', 'y9', 'y10']],
        dpi=100, figsize=(18, 5),
        gridspec_kw={
            "top": 0.9,
            "left": 0.2,
            "right": 0.8,
            "wspace": 0.17,
            "hspace": 0.45,
        },
    )
    fig.suptitle(f'Solution on each edge at t={mesht[t_index]:.02f}', x=0.5, y=1.1, size=15)

    dx = 1.0 / (nx + 1)
    x_nodes = np.linspace(dx, 1.0 - dx, nx)
    
    # Loop over each edge and plot both the exact and computed solutions.
    for e_idx, (s, t2) in enumerate(full_edges.values()):
        offset = e_idx * nx
        offset_end = offset + nx
        na_e_idx = 'y' + str(e_idx + 1)
        ax = axs[na_e_idx]
        ax.plot(x_nodes, y_ex[offset: offset_end], 'k', label='Exact solution', linewidth=4)
        ax.plot(x_nodes, computed_sol[t_index, offset: offset_end], 'r--', label='Computed solution', linewidth=3)
        ax.set_title(f"Edge {e_idx+1}: (v{s+1} → v{t2+1})")
        ax.grid(True)
        ax.set_facecolor('#f5f5f5')  # Light grey background
        # Add legend on one of the subplots.
        if e_idx == 2:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5),
                      ncol=2, fancybox=True, shadow=False, prop={'size': 13})
        ax.set_ylim(-1, 2.3)
        
    # Save the plot if requested.
    if save:
        newpath = check_folder(example)
        name = os.path.join(newpath, f'plot_each_edge_{t_index}.png')
        plt.savefig(name, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
    return None


# =============================================================================
# Function 2: Setup Graph Structure for 3D Visualization
# =============================================================================

def setup_graph_from_full_edges(full_edges):
    """
    Construct a list of edges and corresponding labels for plotting the graph.

    The function converts the full_edges dictionary into a sorted list of edges (with
    vertex names formatted as 'v1', 'v2', etc.) and creates LaTeX-formatted labels.

    Parameters:
    -----------
    full_edges : dict
        Dictionary mapping edge identifiers (e.g., 'e1') to tuples (i, j) of vertex indices.

    Returns:
    --------
    tuple
        (edges, edge_labels):
          - edges: list of tuples, each tuple is (start, end) with vertex names.
          - edge_labels: dictionary mapping each edge tuple to a LaTeX-formatted label.
    """
    edges = []
    edge_labels = {}
    # Sort the keys to maintain order.
    for key in sorted(full_edges.keys(), key=lambda k: int(k[1:])):
        i, j = full_edges[key]
        start, end = f'v{i+1}', f'v{j+1}'
        edge = (start, end)
        edges.append(edge)
        edge_labels[edge] = f'$e_{{{int(key[1:])}}}$'
    return edges, edge_labels


# =============================================================================
# Function 3: Utility Functions for 3D Plotting
# =============================================================================

def euclidean_distance(p1, p2):
    """
    Compute the Euclidean distance between two points.

    Parameters:
    -----------
    p1, p2 : array-like
        Coordinates of the two points.

    Returns:
    --------
    float
        Euclidean distance between p1 and p2.
    """
    return np.linalg.norm(np.array(p2) - np.array(p1))


def plot_curve_on_edge(ax, pos_start, pos_end, z_curve, color, num_points=100, linestyle='solid', linewidth=3, zorder=0):
    """
    Parameterize an edge and plot a curve (e.g., a solution profile) on it.

    The edge is parameterized linearly in R², and the provided z_curve values are used
    to set the z-coordinate for the curve. The result is a 3D curve lying above the edge.

    Parameters:
    -----------
    ax : matplotlib.axes._subplots.Axes3DSubplot
        The 3D axis to plot on.
    pos_start : array-like
        Coordinates of the start vertex (in 2D).
    pos_end : array-like
        Coordinates of the end vertex (in 2D).
    z_curve : array-like
        Array of z-coordinate values corresponding to points along the edge.
    color : str
        Color specification for the curve.
    num_points : int, optional
        Number of points to parameterize along the edge (default is 100).
    linestyle : str, optional
        Line style for the curve (default is 'solid').
    linewidth : int, optional
        Line width (default is 3).
    zorder : int, optional
        Plotting order (default is 0).

    Returns:
    --------
    None
    """
    # Compute the length of the edge.
    a = euclidean_distance(pos_start, pos_end)
    # Parameterize the edge with t in [0, a].
    t = np.linspace(0, a, num_points)
    # Determine the unit direction vector along the edge.
    direction = np.array(pos_end) - np.array(pos_start)
    direction = direction / np.linalg.norm(direction)
    # Compute the x and y coordinates along the edge.
    x_line = pos_start[0] + direction[0] * t
    y_line = pos_start[1] + direction[1] * t
    # Plot the curve with provided z_curve values.
    ax.plot(x_line, y_line, z_curve, color=color, linewidth=linewidth, linestyle=linestyle, zorder=zorder)


# =============================================================================
# Function 4: 3D Graph Plot with Solution Curves
# =============================================================================

def plot_graph_3d_with_curves(computed_sol, exact_sol, full_edges, nx, nt, T, k, positions,
                              show=True, elevation=30, azimuth=30, zorder=0, save=False, example="example_0_3d"):
    """
    Plot the graph in 3D with overlaid solution curves.

    This function visualizes the graph using the vertex positions provided in 'positions'
    and overlays curves corresponding to both the exact and computed solutions along each edge.
    The curves are plotted in the z-direction, and vertical dashed lines connect vertices
    to their solution values.

    Parameters:
    -----------
    computed_sol : np.ndarray
        Computed solution array (nt x total_dof).
    full_edges : dict
        Dictionary mapping edge identifiers to tuples (s, t) of vertex indices.
    nx : int
        Number of interior degrees of freedom per edge.
    nt : int
        Number of time steps.
    T : float
        Final simulation time.
    k : int
        Index in the time mesh at which to visualize the solution.
    positions : dict
        Dictionary mapping vertex names (e.g., 'v1') to their 2D coordinates.
    show : bool, optional
        Whether to display the plot interactively (default is True).
    elevation : float, optional
        Elevation angle for the 3D view (default is 30).
    azimuth : float, optional
        Azimuth angle for the 3D view (default is 30).
    zorder : int, optional
        Z-order for plotting curves (default is 0).
    save : bool, optional
        Whether to save the figure as an image (default is False).
    example : str, optional
        Identifier for saving the plot.

    Returns:
    --------
    None
    """
    # Setup edges and edge labels for plotting.
    edges, edge_labels = setup_graph_from_full_edges(full_edges)

    node_radius = 0.15
    fig = plt.figure(figsize=(12, 6), dpi=100)
    ax = fig.add_subplot(projection='3d')

    # Plot the graph edges as quiver (arrows) to indicate connectivity.
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

    # Plot vertices with different sizes depending on importance.
    bigg_nodes = ['v1', 'v2', 'v3', 'v5', 'v6', 'v8', 'v9', 'v10']
    for node, pos in positions.items():
        x, y = pos
        z = 0
        if node in bigg_nodes:
            ax.scatter(x, y, z, s=50, facecolors='k', edgecolors='black', linewidths=1, alpha=0.4)
        else:
            ax.scatter(x, y, z, s=20, facecolors='k', edgecolors='black', linewidths=1, alpha=0.4)

    # Add edge labels.
    for (start, end) in edges:
        pos_start = np.array(positions[start])
        pos_end = np.array(positions[end])
        midpoint = (pos_start + pos_end) / 2.0
        dx, dy = pos_end - pos_start
        norm = np.hypot(dx, dy)
        if norm == 0:
            continue  # Avoid division by zero.
        perp = np.array([-dy, dx]) / norm
        epsilon = -0.7  # Offset for label placement.
        label_pos = midpoint + epsilon * perp
        label_text = edge_labels.get((start, end), '')
        ax.text(label_pos[0], label_pos[1], 0, label_text, fontsize=12, ha='center', va='center', color='black')

    # Compute the exact solution at time t = mesht[k].
    edges_2, interior_vertices, boundary_vertices = define_graph_full(full_edges)
    all_vertices = interior_vertices + boundary_vertices
    mesht = np.linspace(0, T, nt)
    y_ex = exact_sol(edges_2, all_vertices, interior_vertices, boundary_vertices, nx, t=mesht[k])
    
    # Plot the exact solution curves on each edge.
    for i, (s, t) in enumerate(full_edges.values()):
        e_loc = (f'v{s+1}', f'v{t+1}')
        if e_loc in edges:
            y_loc_ex = y_ex[i * nx: i * nx + nx]
            plot_curve_on_edge(ax, positions[e_loc[0]], positions[e_loc[1]], y_loc_ex,
                               '#153863', num_points=nx, zorder=zorder)
    
    # Plot the computed solution curves (using a dashed line).
    for i, (s, t) in enumerate(full_edges.values()):
        e_loc = (f'v{s+1}', f'v{t+1}')
        if e_loc in edges:
            computed_sol_loc = computed_sol[k][i * nx: i * nx + nx]
            plot_curve_on_edge(ax, positions[e_loc[0]], positions[e_loc[1]], computed_sol_loc,
                               'C3', num_points=nx, linestyle='dashed', linewidth=2, zorder=zorder)

    # Plot vertical dashed lines from the base (z=0) to the vertex solution values.
    for v in all_vertices:
        v_name = f'v{v+1}'
        if v_name in positions:
            E = len(full_edges)
            y_ex_vertex = y_ex[E * nx + v]
            x_point = [positions[v_name][0], positions[v_name][0]]
            y_point = [positions[v_name][1], positions[v_name][1]]
            ax.plot(x_point, y_point, [0, y_ex_vertex], color='k', linewidth=1, linestyle='dashed')

    # Set plot limits and adjust projection scaling.
    ax.set_zlim(-2, 2)
    ax.set_ylim(1, 10)
    ax.set_xlim(-2, 24.5)
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1.5, 1, 0.5, 1]))

    # Set viewing angles.
    ax.view_init(elev=elevation, azim=azimuth)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Create legend for the curves.
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color='#153863', lw=3, linestyle='solid', label='Exact solution'),
        Line2D([0], [0], color='C3', lw=3, linestyle='dashed', label='Computed solution')
    ]
    ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.6, 0.89), ncol=2, fontsize=12)

    # Set the overall title and adjust layout.
    fig.suptitle(f'Real solution vs Computed solution at time t={mesht[k]:.02f}', x=0.55, y=0.89, fontsize=16)
    plt.subplots_adjust(top=0.92)
    
    # Save the figure if requested.
    if save:
        newpath = check_folder(example)
        N_dig = len(str(mesht.shape[0]))
        name = os.path.join(newpath, f'plot_superposition{str(k).zfill(N_dig)}.png')
        plt.savefig(name, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
    return None
