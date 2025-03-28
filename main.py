# =============================================================================
# Import required modules and functions.
# -----------------------------------------------------------------------------
# Graph utilities: define the graph topology (vertices and edges).
from graph import define_graph_full

# Data and PDE functions: coefficients, potentials, initial and boundary conditions.
from data_ex1 import compute_exact_solution    # Exact solution for validation.
from data_ex1 import a_coefficient               # Coefficient function a(x).
from data_ex1 import b_coefficient               # Coefficient function b(x).
from data_ex1 import p_potential                 # Potential function p(x).
from data_ex1 import f_edge                      # Source term or edge function f.
from data_ex1 import initial_condition_dirichlet_full  # Initial condition for Dirichlet BC.
from data_ex1 import g_boundary_condition_ex1         # Dirichlet boundary condition function g.

# PDE solvers: each function implements a different numerical scheme.
from solver import solve_pde_dirichlet_CN    # Crank-Nicolson method.
from solver import solve_pde_dirichlet_IE    # Implicit Euler method.
from solver import solve_pde_dirichlet_EE    # Explicit Euler method (not used here).
from solver import solve_pde_dirichlet_theta # Theta method (generalized time-stepping).
from solver import solve_pde_dirichlet_SIEM  # SIEM method.
from solver import solve_pde_dirichlet_EXPE  # Exponential time integration method.

import numpy as np
import matplotlib.pyplot as plt
import time


# =============================================================================
# Main function: Sets up the graph-based PDE problem, applies different solvers,
# and compares their accuracy against the exact solution.
# -----------------------------------------------------------------------------
def main():
    """
    Sets up the PDE problem on a graph, solves it using various numerical schemes,
    and computes error estimates by comparing with the exact solution.

    Process:
    1. Define the graph structure via edge connectivity.
    2. Set spatial (nx) and temporal (T, nt) discretization parameters.
    3. Load PDE coefficients, potentials, and initial/boundary conditions.
    4. Solve the PDE using different solvers and time the computations.
    5. Compute the exact solution at discrete time steps.
    6. Evaluate the error (max norm) of each numerical solution.
    7. Print the results in a formatted comparison table.

    Inputs:
        - None (configuration and data are imported from external modules)

    Outputs:
        - Console output: graph details, computation times, and error metrics.
    """
    
    # -------------------------------
    # Define the graph structure.
    # -------------------------------
    # 'full_edges' maps edge identifiers to tuples of vertex indices.
    full_edges = {'e1': (0, 2), 'e2': (1, 2), 'e3': (2, 3), 'e4': (3, 4), 'e5': (3, 5),
                  'e6': (4, 6), 'e7': (5, 6), 'e8': (6, 7), 'e9': (7, 8), 'e10': (7, 9)}
    
    # Partition the graph into edges, interior vertices, and boundary vertices.
    edges, interior_vertices, boundary_vertices = define_graph_full(full_edges)
    print("Edges:", edges)
    print("Interior:", interior_vertices)
    print("Boundary:", boundary_vertices)
    
    # Combine all vertices for later use (initial condition computation).
    all_vertices = interior_vertices + boundary_vertices
    
    # -------------------------------
    # Set discretization parameters.
    # -------------------------------
    nx = 100      # Number of spatial discretization points along each edge.
    T = 1.0       # Final time of simulation.
    nt = 500      # Number of time steps.
    
    # -------------------------------
    # Load PDE coefficients and conditions.
    # -------------------------------
    a = a_coefficient              # Coefficient function a(x) in the PDE.
    b = b_coefficient              # Coefficient function b(x) in the PDE.
    p = p_potential                # Potential function p(x).
    f = f_edge                     # Edge source function f.
    # Compute the initial condition on the graph using the full vertex list.
    y0 = initial_condition_dirichlet_full(edges, all_vertices, nx)
    g = g_boundary_condition_ex1   # Dirichlet boundary condition function.
    
    # Aggregate all problem data into a single list for solver input.
    problem_data = [a, b, p, f, y0, g]
    
    # =============================================================================
    # Solve the PDE using various numerical schemes.
    # Each block times the solver, computes the numerical solution, and estimates error.
    # =============================================================================
    
    # --- Implicit Euler (IE) Scheme ---
    start = time.time()  # Start timer for IE solver.
    Y = solve_pde_dirichlet_IE(edges, boundary_vertices, nx, T, nt, problem_data)
    elapsed = time.time() - start  # Compute elapsed time.
    print(f"PDE solved using IE. #dof= {Y.shape[1]}, time= {elapsed:.3f}s")
     
    # Compute the exact solution at each time step for error analysis.
    mesht = np.linspace(0, T, nt)  # Time discretization mesh.
    y_ex = np.array([compute_exact_solution(edges, all_vertices, interior_vertices, boundary_vertices, nx, t=mesht[k])
                     for k in range(nt)])
    
    # Compute maximum relative error (infinity norm) for IE:
    diff = Y - y_ex
    rel_err_IE = np.linalg.norm(diff, axis=1)  # Error at each time step (spatial norm).
    rel_err_IE = np.linalg.norm(rel_err_IE, np.inf)  # Maximum error over time.
    
    # =============================================================================
    # Present the results: Display the error comparison of all solvers.
    # =============================================================================
    print("\nAccuracy:")
    print(f"{'Metric':<7} | {'IE':<9}")
    print(f"{'-'*7} | {'-'*9}")
    print(f"{'Error':<7} | {rel_err_IE:<9.3e}")


    # -----------------------------------------------------------------------------
    # Import plotting functions and position definitions for visualization.
    # -----------------------------------------------------------------------------
    from data_ex1 import define_positions_example_1  # Provides vertex positions for plotting.
    from plot_functions import plot_edges            # Function to plot the solution along each edge.
    from plot_functions import plot_graph_3d_with_curves  # Function to plot a 3D graph with solution curves.
    # -----------------------------------------------------------------------------
    # Plot the solution along each edge.
    # -----------------------------------------------------------------------------
    print(f"\nPlotting the solution per edge...")
    t_index = int(nt/3) + 1

    # plot_edges: Visualizes the solution (Y) on each edge of the graph.
    # Inputs:
    #   - Y: Solution array.
    #   - compute_exact_solution: function exact solution.
    #   - nx, nt, T: Spatial and temporal discretization parameters.
    #   - t_index: Time index used for plotting the solution curve.
    #   - full_edges: The dictionary defining graph connectivity.
    #   - show: Whether to display the plot.
    #   - save: Whether to save the plot to file.
    #   - example: Identifier for the saved file.
    plot_edges(Y, compute_exact_solution, nx, nt, T, t_index,
                full_edges, show=True, save=False, example='example_0')

    # -----------------------------------------------------------------------------
    # Plot the graph along with the solution curves in 3D.
    # -----------------------------------------------------------------------------
    # Obtain vertex positions from a predefined example.
    positions = define_positions_example_1()
    # Select a specific time index to visualize the solution curve on the graph.
    t_index = int(nt/3) + 1

    print(f"\nPlotting graph and the solution on it...")
    # plot_graph_3d_with_curves: Generates a 3D visualization of the graph,
    # overlaying the solution curves. Inputs include:
    #   - Y: The solution array.
    #   - compute_exact_solution: function exact solution.
    #   - full_edges: Graph connectivity dictionary.
    #   - nx, nt, T: Discretization parameters.
    #   - t_index: Time index used for plotting the solution curve.
    #   - positions: Vertex positions.
    #   - show: Whether to display the plot.
    #   - elevation, azimuth: Viewing angles for the 3D plot.
    #   - zorder: Plot layering order.
    #   - save: Whether to save the plot to file.
    #   - example: Identifier for the saved file.
    plot_graph_3d_with_curves(Y, compute_exact_solution, full_edges, nx, nt, T, t_index, positions, 
                            show=True, elevation=33, azimuth=320, zorder=100, 
                            save=False, example="example_0_3d")

# -----------------------------------------------------------------------------
# Run the main function if this script is executed as the main module.
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
