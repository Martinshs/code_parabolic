

def adjacent_vertices(full_edges, v):
    list_vertices = set()
    for i, (vi,vo) in enumerate(full_edges.values()): 
        if v == vi or v == vo:
            list_vertices.add(vi)
            list_vertices.add(vo)
    return list_vertices

def define_graph_full(full_edges):
    """
     desceiption
    """
    edges = full_edges.values()
    allv= set()
    for (s,t) in edges:
        allv.add(s)
        allv.add(t)
    all_vertices= sorted(list(allv))  # should be [0..10]
    
    interior_vertices = []
    for v in all_vertices:
        count = 0
        for (s,t) in edges:
            if v==s or v==t:
                count+=1
                adj_vert = adjacent_vertices(full_edges, v)
                if count==2 and adj_vert.intersection(all_vertices) == adj_vert: #The interior verices conect two edges and has the same degree that in the full graph
                    interior_vertices.append(v)
    boundary_vertices= [v for v in all_vertices if v not in interior_vertices]
    edges = list(edges)

    return edges, interior_vertices, boundary_vertices