def adjacent_vertices(full_edges, v):
    """
    Compute the set of vertices adjacent to a given vertex in the graph.

    Parameters:
    -----------
    full_edges : dict
        A dictionary where keys are edge identifiers and values are tuples (s, t)
        representing the endpoints of each edge.
    v : int or any hashable type
        The vertex for which the adjacent vertices are to be found.

    Returns:
    --------
    set
        A set containing all vertices that share an edge with vertex v.
    """
    list_vertices = set()
    # Iterate over each edge in the graph.
    for i, (vi, vo) in enumerate(full_edges.values()):
        # If the vertex 'v' is one of the endpoints, add both endpoints to the set.
        if v == vi or v == vo:
            list_vertices.add(vi)
            list_vertices.add(vo)
    return list_vertices


def define_graph_full(full_edges):
    """
    Construct a graph representation from a dictionary of edges and classify the vertices 
    as interior or boundary vertices.

    The function processes the given edge definitions to determine:
      - The list of edges.
      - The interior vertices: vertices that are incident to exactly two edges and whose 
        set of adjacent vertices (neighbors) is completely contained within the overall vertex set.
      - The boundary vertices: vertices not classified as interior.

    Parameters:
    -----------
    full_edges : dict
        A dictionary where keys are edge identifiers and values are tuples (s, t)
        representing the connectivity between vertices in the graph.

    Returns:
    --------
    tuple
        A tuple (edges, interior_vertices, boundary_vertices) where:
          - edges: A list of edge tuples extracted from full_edges.
          - interior_vertices: A list of vertices considered as 'interior' based on their connectivity.
          - boundary_vertices: A list of vertices that are not classified as interior.
    """
    # Extract edge tuples from the dictionary.
    edges = full_edges.values()
    allv = set()
    # Collect all unique vertices from all edges.
    for (s, t) in edges:
        allv.add(s)
        allv.add(t)
    # Sort vertices to maintain a consistent ordering.
    all_vertices = sorted(list(allv))
    
    interior_vertices = []
    # Loop over each vertex to determine if it is an interior vertex.
    for v in all_vertices:
        count = 0  # Counter for the number of edges incident to vertex v.
        for (s, t) in edges:
            if v == s or v == t:
                count += 1
                # Compute the adjacent vertices for v.
                adj_vert = adjacent_vertices(full_edges, v)
                # Check if v is connected to exactly two edges and if all its neighbors
                # are within the overall set of vertices. This condition is used to
                # heuristically classify v as an interior vertex.
                if count == 2 and adj_vert.intersection(all_vertices) == adj_vert:
                    interior_vertices.append(v)
    # Boundary vertices are those vertices not classified as interior.
    boundary_vertices = [v for v in all_vertices if v not in interior_vertices]
    # Convert edges from dict_values to a list for further processing.
    edges = list(edges)

    return edges, interior_vertices, boundary_vertices
