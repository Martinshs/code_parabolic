import numpy as np


from numpy.linalg import inv

from build_matrices import buildFEM_matrices
from build_matrices import apply_dirichlet_bc
from build_load import compute_load_vector

from data_ex1 import initial_condition_dirichlet_full


def solve_pde_dirichlet_IE(edges, interior_vertices, boundary_vertices, nx, T, nt):
    """
    We'll build M,A,B,C for all vertices => E*nx + #all_vertices dof.
    Then at each time step, we do row modifications for boundary dofs => y(b)= e^-t.
    Meanwhile, the *initial condition* at each vertex dof is the exact solution
    at t=0 for that vertex, determined by which edge sub-interval the vertex is on.
    """
    E= len(edges)
    dx= 1.0/(nx+1)

    # Gather all vertices in the graph
    allv= set()
    for (s,t) in edges:
        allv.add(s)
        allv.add(t)
    all_vertices= sorted(list(allv))

    # 1) Build global matrices
    M_H, A_H, B_H, C_H, dof_edge, dof_vertex= buildFEM_matrices(edges, nx, all_vertices)
    total_dof= M_H.shape[0]

    # 2) Build initial condition y0
    y0 = initial_condition_dirichlet_full(edges, all_vertices,nx)

    # 3) Time stepping
    dt= T/ nt
    times= np.linspace(0, T, nt)
    Y= np.zeros((nt, total_dof))
    Y[0,:]= y0

    for n in range(nt-1):
        t_n_plus_1= times[n+1]
        # build load
        F= compute_load_vector(edges, nx, all_vertices, t_n_plus_1)
        # system => (M + dt(A+B+C)) => ...
        A_sys= M_H + dt*( A_H + B_H + C_H )
        RHS= M_H @ Y[n,:] + dt* F

        # impose Dirichlet => y(b, t)= e^-t on boundary vertices
        A_sys, RHS= apply_dirichlet_bc(A_sys, RHS, boundary_vertices, all_vertices, t_n_plus_1, E, nx)

        # solve
        try:
            y_next= inv(A_sys)@ RHS
        except np.linalg.LinAlgError:
            raise ValueError("Singular or ill-conditioned system for Dirichlet BC!")
        Y[n+1,:]= y_next

    return times, Y, all_vertices



def solve_pde_dirichlet_CN(edges, interior_vertices, boundary_vertices, nx, T, nt):
    """
    Solve the PDE with Dirichlet BC using the Crank–Nicolson scheme.
    Builds global matrices for all vertices (E*nx + number of vertices dof).
    The initial condition is the exact solution at t=0 on each vertex.
    """
    E = len(edges)
    dx = 1.0/(nx+1)

    # Gather all vertices in the graph
    allv = set()
    for (s, t) in edges:
        allv.add(s)
        allv.add(t)
    all_vertices = sorted(list(allv))

    # 1) Build global matrices
    M_H, A_H, B_H, C_H, dof_edge, dof_vertex = buildFEM_matrices(edges, nx, all_vertices)
    total_dof = M_H.shape[0]

    # 2) Build initial condition y0
    y0 = initial_condition_dirichlet_full(edges, all_vertices, nx)

    # 3) Time stepping
    dt = T / nt
    times = np.linspace(0, T, nt)
    Y = np.zeros((nt, total_dof))
    Y[0, :] = y0

    # Precompute the constant matrices for the Crank–Nicolson scheme
    LHS_matrix = M_H + (dt/2) * (A_H + B_H + C_H)
    RHS_matrix = M_H - (dt/2) * (A_H + B_H + C_H)

    for n in range(nt-1):
        t_n = times[n]
        t_n_plus_1 = times[n+1]

        # Compute load vectors at current and next time
        F_n = compute_load_vector(edges, nx, all_vertices, t_n)
        F_np1 = compute_load_vector(edges, nx, all_vertices, t_n_plus_1)

        # Assemble the right-hand side vector for Crank-Nicolson
        RHS = RHS_matrix @ Y[n, :] + (dt/2) * (F_n + F_np1)

        # Apply Dirichlet boundary conditions at time t_n+1
        # This enforces y(boundary, t_{n+1}) = e^{-t} (or your prescribed BC function)
        A_sys, RHS = apply_dirichlet_bc(LHS_matrix, RHS,
                                        boundary_vertices,
                                        all_vertices,
                                        t_n_plus_1, E, nx)
        

        # Solve for the next time step
        try:
            y_next = np.linalg.inv(A_sys) @ RHS
        except np.linalg.LinAlgError:
            raise ValueError("Singular or ill-conditioned system for Dirichlet BC!")
        Y[n+1, :] = y_next

    return times, Y, all_vertices



from scipy.linalg import expm

def check_CFL_EE(M, A, B, C, dt):
    r"""
    For the Explicit Euler scheme applied to
    \[
    M \dot{y} = - (A+B+C) y,
    \]
    stability for the scalar test problem \(y' = -\lambda y\) requires
    \[
    \Delta t < \frac{2}{\lambda}.
    \]
    In the matrix case we demand
    \[
    \Delta t < \frac{2}{\lambda_{\max}(M^{-1}(A+B+C))}.
    \]
    """
    L = np.linalg.inv(M) @ (A+B+C)
    eigenvalues = np.linalg.eigvals(L)
    lambda_max = np.max(np.real(eigenvalues))
    dt_bound = 2.0 / lambda_max
    if dt < dt_bound:
        return dt, True
    else:
        return dt_bound, False

def check_CFL_theta(M, A, B, C, dt, theta):
    r"""
    For the \(\theta\)-method applied to
    \[
    M \dot{y} = - (A+B+C) y,
    \]
    when \(\theta < \frac{1}{2}\) one obtains the conditional stability requirement
    \[
    \Delta t < \frac{2}{(1-2\theta)\,\lambda_{\max}(M^{-1}(A+B+C))}.
    \]
    For \(\theta\ge\frac{1}{2}\) (A-stable) no CFL restriction applies.
    """
    if theta >= 0.5:
        return dt, True  # A-stable: no restriction.
    L = np.linalg.inv(M) @ (A+B+C)
    eigenvalues = np.linalg.eigvals(L)
    lambda_max = np.max(np.real(eigenvalues))
    dt_bound = 2.0 / ((1 - 2*theta) * lambda_max)
    if dt < dt_bound:
        return dt, True
    else:
        return dt_bound, False

def check_CFL_SIE(M, A, B, C, dt):
    r"""
    For the Semi-Implicit Euler method, where the implicit part is
    \(A\) and the explicit part is \(B+C\), the explicit update is stable if
    \[
    \Delta t < \frac{2}{\lambda_{\max}(M^{-1}(B+C))}.
    \]
    """
    L_explicit = np.linalg.inv(M) @ (B+C)
    eigenvalues = np.linalg.eigvals(L_explicit)
    lambda_max = np.max(np.real(eigenvalues))
    dt_bound = 2.0 / lambda_max
    if dt < dt_bound:
        return dt, True
    else:
        return dt_bound, False

def check_CFL_EXPE(M, A, B, C, dt):
    r"""
    Although the Exponential Euler method is unconditionally stable,
    accuracy requires \(\Delta t\,\lambda_{\max}(L) < 1\) with
    \[
    L = - M^{-1}(A+B+C).
    \]
    Thus, one recommends
    \[
    \Delta t < \frac{1}{\lambda_{\max}(L)}.
    \]
    """
    L = -np.linalg.inv(M) @ (A+B+C)
    eigenvalues = np.linalg.eigvals(L)
    lambda_max = np.max(np.abs(np.real(eigenvalues)))
    dt_bound = 1.0 / lambda_max
    if dt < dt_bound:
        return dt, True
    else:
        return dt_bound, False

# ----------------------------------------------------------------------
# 1. Explicit Euler Method
# ----------------------------------------------------------------------
def solve_pde_dirichlet_EE(edges, interior_vertices, boundary_vertices, nx, T, nt):
    """
    Solve the parabolic PDE with Dirichlet BCs using the Explicit Euler method.
    
    Update: 
      \( y^{n+1} = y^n + \Delta t \,M^{-1}\Bigl(F - (A+B+C)y^n\Bigr) \).
    
    A CFL check is performed.
    """
    E = len(edges)
    dx = 1.0 / (nx + 1)
    
    # Gather vertices.
    allv = set()
    for (s,t) in edges:
        allv.add(s)
        allv.add(t)
    all_vertices = sorted(list(allv))
    
    # Build global FEM matrices.
    M_H, A_H, B_H, C_H, dof_edge, dof_vertex = buildFEM_matrices(edges, nx, all_vertices)
    total_dof = M_H.shape[0]
    
    # Initial condition.
    y0 = initial_condition_dirichlet_full(edges, all_vertices, nx)
    
    dt = T / nt
    times = np.linspace(0, T, nt)
    Y = np.zeros((nt, total_dof))
    Y[0, :] = y0
    
    # Check CFL for Explicit Euler.
    dt_check, is_stable = check_CFL_EE(M_H, A_H, B_H, C_H, dt)
    if not is_stable:
        raise ValueError(f"Explicit Euler: dt = {dt:.3e} exceeds CFL limit; recommend dt = {dt_check:.3e}.")
    else:
        print(f"CFL condition satisfy. Maximum possible dt = {dt_check:.3e} to satisfy the CFL condition.")
    for n in range(nt-1):
        t_next = times[n+1]
        # Compute load vector at time t_next.
        F = compute_load_vector(edges, nx, all_vertices, t_next)
        # Explicit Euler update: compute the right-hand side.
        # Note: M_H @ y^n + dt*(F - (A_H+B_H+C_H) y^n)
        RHS = M_H @ Y[n, :] + dt * (F - (A_H + B_H + C_H) @ Y[n, :])
        A_sys = M_H.copy()  # For explicit update, the "system" is M_H.
        
        # Impose Dirichlet conditions: here boundary_vertices holds the indices.
        A_sys, RHS = apply_dirichlet_bc(A_sys, RHS, boundary_vertices, all_vertices, t_next, E, nx)
        try:
            y_next = np.linalg.inv(A_sys) @ RHS
        except np.linalg.LinAlgError:
            raise ValueError("Singular or ill-conditioned system in Explicit Euler!")
        Y[n+1, :] = y_next
        
    return times, Y, all_vertices

# ----------------------------------------------------------------------
# 2. Theta-Method
# ----------------------------------------------------------------------
def solve_pde_dirichlet_theta(edges, interior_vertices, boundary_vertices, nx, T, nt, theta):
    """
    Solve the parabolic PDE with Dirichlet BCs using the \(\theta\)-method.
    
    Update:
      \([M + \theta\,\Delta t\,(A+B+C)]\,y^{n+1} = M\,y^n + \Delta t \Bigl[(1-\theta)(F-(A+B+C)y^n) + \theta\,F\Bigr]\).
    
    A stability/CFL check is performed for \(\theta < 1/2\).
    """
    E = len(edges)
    dx = 1.0 / (nx + 1)
    
    # Gather vertices.
    allv = set()
    for (s,t) in edges:
        allv.add(s)
        allv.add(t)
    all_vertices = sorted(list(allv))
    
    # Build global FEM matrices.
    M_H, A_H, B_H, C_H, dof_edge, dof_vertex = buildFEM_matrices(edges, nx, all_vertices)
    total_dof = M_H.shape[0]
    
    # Initial condition.
    y0 = initial_condition_dirichlet_full(edges, all_vertices, nx)
    
    dt = T / nt
    times = np.linspace(0, T, nt)
    Y = np.zeros((nt, total_dof))
    Y[0, :] = y0
    
    # Check CFL for theta-method (if theta < 0.5).
    dt_check, is_stable = check_CFL_theta(M_H, A_H, B_H, C_H, dt, theta)
    if not is_stable:
        print(f"Theta-method: dt = {dt:.3e} exceeds stability limit; recommend dt = {dt_check:.3e}.")
    
    for n in range(nt-1):
        t_next = times[n+1]
        F = compute_load_vector(edges, nx, all_vertices, t_next)
        # System matrix and RHS for the theta-method.
        A_sys = M_H + dt * theta * (A_H + B_H + C_H)
        RHS = M_H @ Y[n, :] + dt * ( (1 - theta) * (F - (A_H+B_H+C_H) @ Y[n, :]) + theta * F )
        
        A_sys, RHS = apply_dirichlet_bc(A_sys, RHS, boundary_vertices, all_vertices, t_next, E, nx)
        try:
            y_next = np.linalg.inv(A_sys) @ RHS
        except np.linalg.LinAlgError:
            raise ValueError("Singular or ill-conditioned system in Theta-method!")
        Y[n+1, :] = y_next
        
    return times, Y, all_vertices

# ----------------------------------------------------------------------
# 3. Semi-Implicit Euler Method
# ----------------------------------------------------------------------
def solve_pde_dirichlet_SIEM(edges, interior_vertices, boundary_vertices, nx, T, nt):
    """
    Solve the parabolic PDE with Dirichlet BCs using the Semi-Implicit Euler method.
    
    Here, the operator \(A\) is treated implicitly while \(B+C\) is treated explicitly:
      \([M + \Delta t\,A]\,y^{n+1} = M\,y^n + \Delta t\,(F - (B+C)y^n)\).
    
    A CFL check is performed on the explicit part.
    """
    E = len(edges)
    dx = 1.0 / (nx + 1)
    
    # Gather vertices.
    allv = set()
    for (s,t) in edges:
        allv.add(s)
        allv.add(t)
    all_vertices = sorted(list(allv))
    
    # Build global FEM matrices.
    M_H, A_H, B_H, C_H, dof_edge, dof_vertex = buildFEM_matrices(edges, nx, all_vertices)
    total_dof = M_H.shape[0]
    
    # Initial condition.
    y0 = initial_condition_dirichlet_full(edges, all_vertices, nx)
    
    dt = T / nt
    times = np.linspace(0, T, nt)
    Y = np.zeros((nt, total_dof))
    Y[0, :] = y0
    
    # Check CFL for the explicit part of Semi-Implicit Euler.
    dt_check, is_stable = check_CFL_SIE(M_H, A_H, B_H, C_H, dt)
    if not is_stable:
        print(f"Semi-Implicit Euler: dt = {dt:.3e} exceeds stability limit; recommend dt = {dt_check:.3e}.")
    
    for n in range(nt-1):
        t_next = times[n+1]
        F = compute_load_vector(edges, nx, all_vertices, t_next)
        A_sys = M_H + dt * A_H
        RHS = M_H @ Y[n, :] + dt * (F - (B_H + C_H) @ Y[n, :])
        
        A_sys, RHS = apply_dirichlet_bc(A_sys, RHS, boundary_vertices, all_vertices, t_next, E, nx)
        try:
            y_next = np.linalg.inv(A_sys) @ RHS
        except np.linalg.LinAlgError:
            raise ValueError("Singular or ill-conditioned system in Semi-Implicit Euler!")
        Y[n+1, :] = y_next
        
    return times, Y, all_vertices

# ----------------------------------------------------------------------
# 4. Exponential Euler Method
# ----------------------------------------------------------------------
def solve_pde_dirichlet_EXPE(edges, interior_vertices, boundary_vertices, nx, T, nt):
    """
    Solve the parabolic PDE with Dirichlet BCs using the Exponential Euler method.
    
    The update is:
      \[
      y^{n+1} = e^{\Delta t\,L}\,y^n + \Delta t\,\varphi_1(\Delta t\,L)\,M^{-1}F,
      \]
    with
      \[
      L = -M^{-1}(A+B+C) \quad \text{and} \quad \varphi_1(z)=\frac{e^z-1}{z}.
      \]
    An accuracy condition is checked.
    """
    E = len(edges)
    dx = 1.0 / (nx + 1)
    
    # Gather vertices.
    allv = set()
    for (s,t) in edges:
        allv.add(s)
        allv.add(t)
    all_vertices = sorted(list(allv))
    
    # Build global FEM matrices.
    M_H, A_H, B_H, C_H, dof_edge, dof_vertex = buildFEM_matrices(edges, nx, all_vertices)
    total_dof = M_H.shape[0]
    
    # Initial condition.
    y0 = initial_condition_dirichlet_full(edges, all_vertices, nx)
    
    dt = T / nt
    times = np.linspace(0, T, nt)
    Y = np.zeros((nt, total_dof))
    Y[0, :] = y0
    
    # Check accuracy condition for Exponential Euler.
    dt_check, is_stable = check_CFL_EXPE(M_H, A_H, B_H, C_H, dt)
    if not is_stable:
        print(f"Exponential Euler: dt = {dt:.3e} exceeds accuracy limit; recommend dt = {dt_check:.3e}.")
    
    for n in range(nt-1):
        t_next = times[n+1]
        F = compute_load_vector(edges, nx, all_vertices, t_next)
        # Define L = - M_H^{-1}(A_H+B_H+C_H).
        L = -np.linalg.inv(M_H) @ (A_H+B_H+C_H)
        exp_dtL = expm(dt * L)
        # Compute phi1(dtL) = (exp(dtL)-I)/(dtL)
        phi1_dtL = np.linalg.solve(dt * L, (exp_dtL - np.eye(L.shape[0])))
        y_next = exp_dtL @ Y[n, :] + dt * phi1_dtL @ (np.linalg.inv(M_H) @ F)
        
        # Enforce Dirichlet conditions a posteriori.
        A_dummy = np.eye(M_H.shape[0])
        RHS_dummy = y_next.copy()
        A_dummy, RHS_dummy = apply_dirichlet_bc(A_dummy, RHS_dummy, boundary_vertices, all_vertices, t_next, E, nx)
        y_next = RHS_dummy
        
        Y[n+1, :] = y_next
        
    return times, Y, all_vertices
