
from graph import define_graph_full
from data_ex1 import compute_exact_solution

from solver import solve_pde_dirichlet_CN
from solver import solve_pde_dirichlet_IE
from solver import solve_pde_dirichlet_EE
from solver import solve_pde_dirichlet_theta
from solver import solve_pde_dirichlet_SIEM
from solver import solve_pde_dirichlet_EXPE

import numpy as np
import matplotlib.pyplot as plt
import time
#from memory_profiler import memory_usage

def main():


    full_edges = {'e1' :(0,2), 'e2' :(1,2), 'e3' :(2,3), 'e4' :(3,4), 'e5' :(3,5),
                'e6' :(4,6), 'e7' :(5,6), 'e8' :(6,7), 'e9' :(7,8), 'e10' :(7,9)}

    edges, interior_vertices, boundary_vertices= define_graph_full(full_edges)
    print("Edges:", edges)
    print("Interior:", interior_vertices)
    print("Boundary:", boundary_vertices)

    nx= 10
    T= 1.0
    nt= 1500

    start= time.time()
    times, Y, all_vertices= solve_pde_dirichlet_IE(edges, interior_vertices, boundary_vertices, nx, T, nt)
    elapsed= time.time()- start
    print(f"Solved PDE using IE. #dof= {Y.shape[1]}, time= {elapsed:.3f}s")
     
    mesht = np.linspace(0,T,nt)
    y_ex= np.array([compute_exact_solution(edges, all_vertices,interior_vertices, boundary_vertices, nx, t=  mesht[k]) for k in range(nt)])
    
    diff= Y-y_ex
    rel_err = np.linalg.norm(diff, axis=1)
    rel_err = np.linalg.norm(rel_err, np.inf)
    print(f"Relative error sup in time IE => {rel_err:.3e}")

    
    start= time.time()
    times2, Y2, all_vertices2= solve_pde_dirichlet_CN(edges, interior_vertices, boundary_vertices, nx, T, nt)
    elapsed= time.time()- start
    print(f"Solved PDE using CN.  #dof= {Y.shape[1]}, time= {elapsed:.3f}s")
    
    diff= Y2-y_ex
    rel_err = np.linalg.norm(diff, axis=1)
    rel_err = np.linalg.norm(rel_err, np.inf)
    print(f"Relative error sup in time CN => {rel_err:.3e}")

    
    start= time.time()
    times3, Y3, all_vertices2= solve_pde_dirichlet_EE(edges, interior_vertices, boundary_vertices, nx, T, nt)
    elapsed= time.time()- start
    print(f"Solved PDE using EE.  #dof= {Y.shape[1]}, time= {elapsed:.3f}s")
    
    diff= Y3-y_ex
    rel_err = np.linalg.norm(diff, axis=1)
    rel_err = np.linalg.norm(rel_err, np.inf)
    print(f"Relative error sup in time EE => {rel_err:.3e}")

    start= time.time()
    times4, Y4, all_vertices2= solve_pde_dirichlet_theta(edges, interior_vertices, boundary_vertices, nx, T, nt, 1/4)
    elapsed= time.time()- start
    print(f"Solved PDE using theta (1/4).  #dof= {Y.shape[1]}, time= {elapsed:.3f}s")

    diff= Y4-y_ex
    rel_err = np.linalg.norm(diff, axis=1)
    rel_err = np.linalg.norm(rel_err, np.inf)
    print(f"Relative error sup in time theta (1/4) => {rel_err:.3e}")

    start= time.time()
    times5, Y5, all_vertices2 = solve_pde_dirichlet_theta(edges, interior_vertices, boundary_vertices, nx, T, nt, 3/4)
    elapsed= time.time()- start
    print(f"Solved PDE using theta (3/4).  #dof= {Y.shape[1]}, time= {elapsed:.3f}s")

    diff= Y5-y_ex
    rel_err = np.linalg.norm(diff, axis=1)
    rel_err = np.linalg.norm(rel_err, np.inf)
    print(f"Relative error sup in time theta (3/4) => {rel_err:.3e}")


    start= time.time()
    times5, Y6, all_vertices2 = solve_pde_dirichlet_SIEM(edges, interior_vertices, boundary_vertices, nx, T, nt)
    elapsed= time.time()- start
    print(f"Solved PDE using SIEM. #dof= {Y.shape[1]}, time= {elapsed:.3f}s")

    diff= Y6-y_ex
    rel_err = np.linalg.norm(diff, axis=1)
    rel_err = np.linalg.norm(rel_err, np.inf)
    print(f"Relative error sup in time SIEM => {rel_err:.3e}")
    

    start= time.time()
    times5, Y7, all_vertices2 = solve_pde_dirichlet_EXPE(edges, interior_vertices, boundary_vertices, nx, T, nt)
    elapsed= time.time()- start
    print(f"Solved PDE using EXPE. #dof= {Y.shape[1]}, time= {elapsed:.3f}s")

    diff= Y7-y_ex
    rel_err = np.linalg.norm(diff, axis=1)
    rel_err = np.linalg.norm(rel_err, np.inf)
    print(f"Relative error sup in time EXPE => {rel_err:.3e}")


if __name__=="__main__":
    main()
