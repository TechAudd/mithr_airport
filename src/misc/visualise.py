import networkx as nx
import matplotlib.pyplot as plt

def visualize_workflow(workflow):    
    graph = workflow.get_graph()
    G = nx.DiGraph()
    for node in graph.nodes:
        G.add_node(node)
    for edge in graph.edges:
        G.add_edge(edge[0], edge[1])

    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightblue", font_size=10, font_weight="bold", arrows=True)
    plt.title("Workflow Graph")
    plt.show()

def generate_mermaid_code(workflow):
    mermaid_code = workflow.get_graph().draw_mermaid()
    lines = mermaid_code.splitlines()
    filtered_lines = [line for line in lines if 'classDef' not in line]
    filtered_lines = [line for line in filtered_lines if 'linear' not in line]
    mermaid_code = "\n".join(filtered_lines)
    mermaid_code += "\n\tclassDef default fill:#00000000,line-height:1.2\n\tclassDef first fill:#00ff0055\n\tclassDef last fill:#ff0000"
    return mermaid_code
