import networkx as nx

def build_dependency_graph():
    graph = nx.DiGraph()
    graph.add_edge("Input", "Operation")
    graph.add_edge("Operation", "Output")
    return graph
