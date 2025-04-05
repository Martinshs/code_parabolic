# PDE on Graphs

This repository provides a library to solve parabolic partial differential equations (PDEs) of the type

$$\partial_t y_e - \partial_x(a \partial_x y_e) + b \partial_x y_e + p y_e = f_e,$$

on finite metric graphs with Dirichlet boundary conditions at the boundary vertices. The implementation is based on a finite element discretization that takes into account both the interior and boundary degrees of freedom of a graph. The code was constructed to ease adaptation for any graph.

Several schemes can be used for the time integration, including Implicit Euler, Crank–Nicolson, Explicit Euler, the θ-method, Semi-Implicit Euler, and Exponential Euler.
<p align="center">
<img src="/images/images_example_3d/plot_superposition167.png" alt="Gráfico" width="400"/>
</p>
---

## Problem Formulation

Consider a connected finite metric graph $\mathcal{G} := (\mathcal{E}, \mathcal{V})$ with vertices $v \in \mathcal{V}$ and edges $e \in \mathcal{E}$. For each edge $e \in \mathcal{E}$, the incidence vector function $n_e: \mathcal{V} \to \{-1, 0, 1\}$ is defined by

$$
n_e(v) = 
\begin{cases} 
-1 & \text{if } e \text{ starts at } v, \\
1 & \text{if } e \text{ ends at } v, \\
0 & \text{otherwise}.
\end{cases}
$$

This function encapsulates the directional relationship of edges with respect to vertices in the graph. Let $T > 0$; over each edge $e \in \mathcal{E}$, we consider the heat equation

$$
\begin{cases}
\partial_t y_e - \partial_{x}\bigl(a_e \partial_{x} y_e\bigr) + b_e \partial_x y_e + p_e y_e = f_e, & (x,t) \in e \times (0, T), \\
y_e(x, 0) = y_e^0(x), & x \in e,
\end{cases}
$$

complemented with coupling and boundary conditions:

$$\begin{cases}
y_{e_1}(v,t) = y_{e_2}(v,t) & \forall e_1,e_2 \in \mathcal{E}(v),\hspace{1mm} (v,t)\in \mathcal{V}_0 \times (0,T), \qquad &\text{(Continuity condition)} \newline
y_e(v,t) = g_v(t) &  (v,t)\in\mathcal{V}_b \times (0,T), \qquad  &\text{(Boundary condition)} \newline
\sum \partial_x y_e(v,t) \, n_e(v) = 0 & (v,t)\in \mathcal{V}_0 \times (0,T), \qquad  &\text{(Kirchhoff condition)}
\end{cases}$$


---

## Repository Structure

```
├── data_ex1.py             # Coefficient functions, exact solution definitions, initial and boundary conditions.
├── build_matrices.py       # Construction of local and global FEM matrices for edges and vertices.
├── build_load.py           # Assembly of the global load vector.
├── graph.py                # Graph structure definition and related utilities.
├── solver.py               # Implementation of various time-stepping solvers (IE, CN, EE, θ-method, SIEM, EXPE).
├── plot_functions.py       # Functions for visualizing 2D edge solutions and 3D graph plots.
├── utils.py                # Utility functions (e.g., folder checking for saving images).
├── notebooks/              # IPython notebook illustrating usage of the coder
|                           # IPython notebook analizing error and performance.
└── README.md               # This file.
```

---

## Features

- Finite Element Discretization on Graphs
- Multiple Time-Stepping Schemes
- CFL Checks for Explicit Schemes
- Visualization Tools
- Exact Solution Comparison

---

## Requirements

- Python 3.x
- NumPy
- SciPy
- Matplotlib

---

## Usage

You can run the solvers from the notebooks or via Python scripts. A more detail description of the usage of the code can be found in the Usage_Tutorial.ipynb.

---

## Results

Includes detailed visualizations (mosaic and 3D) to evaluate solver performance and solution accuracy. 
<p align="center">
<img src="/images/images_example_2d/plot_each_edge_167.png" alt="Gráfico" width="800"/>
</p>


---

## Recommendations & Future Work

- Edges with different lengths and different spatial discretizations ($dx$) per edge 
- Different coefficients per edge
- Adaptation to Robin and Neumann boundary conditions


---

## License

This project is licensed under the **MIT License**.

---

## Acknowledgements

Founded by the [TRR154](https://www.trr154.fau.de/trr-154-en/).

