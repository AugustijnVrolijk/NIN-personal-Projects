import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import itertools
import torch

pixels = 12
neurons = 6
features = 2

subset_sizes = [pixels, neurons, features]
subset_color = ["gold", "violet", "limegreen"]


input1 = np.random.random((neurons,features))
print(input)
input2 = np.random.random((neurons, pixels))


Graph = nx.Graph()

for n in range(neurons):
    nName = f"n{n}"
    for f in range(features):
        fName = f"f{f}"   
        Graph.add_edge(nName, fName, weight=input1[n, f])
    for p in range(pixels):
        pName = f"p{p}"   
        Graph.add_edge(nName, pName, weight=input2[n, p])

def multilayered_graph(*subset_sizes):
    extents = nx.utils.pairwise(itertools.accumulate((0,) + subset_sizes))
    layers = [range(start, end) for start, end in extents]
    G = nx.Graph()
    for i, layer in enumerate(layers):
        G.add_nodes_from(layer, layer=i)
    for layer1, layer2 in nx.utils.pairwise(layers):
        G.add_edges_from(itertools.product(layer1, layer2))
    return G

def plotMultilayer():
    G = multilayered_graph(*subset_sizes)
    color = [subset_color[data["layer"]] for v, data in G.nodes(data=True)]
    pos = nx.multipartite_layout(G, subset_key="layer")
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, node_color=color, with_labels=False)
    plt.axis("equal")
    plt.show()



def plotSpringLayout(graph):

    pos = nx.spring_layout(Graph, seed=7)  # positions for all nodes - seed for reproducibility
    elarge = [(u, v) for (u, v, d) in Graph.edges(data=True) if d["weight"] > 0.5]
    esmall = [(u, v) for (u, v, d) in Graph.edges(data=True) if d["weight"] <= 0.5]



    # nodes
    nx.draw_networkx_nodes(Graph, pos, node_size=700)

    # edges
    nx.draw_networkx_edges(Graph, pos, edgelist=elarge, width=1)
    nx.draw_networkx_edges(
        Graph, pos, edgelist=esmall, width=1, alpha=0.5, edge_color="b", style="dashed"
    )

    # node labels
    nx.draw_networkx_labels(Graph, pos, font_size=20, font_family="sans-serif")
    # edge weight labels
    #edge_labels = nx.get_edge_attributes(Graph, "weight")
    #nx.draw_networkx_edge_labels(Graph, pos, edge_labels)

    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


plotSpringLayout(Graph)


class GraphKeyIterator:
    def __init__(self):
        # Initialize with the letter 'A' (ASCII value of 65)
        self.currSymbol = ord('A')
        self.curr = 0

    def __iter__(self):
        # This makes the object itself an iterator
        return self

    def getIter(self):
        return self.curr

    def getLabel(self):
        return self.currSymbol

    def __next__(self):
        # Check if we are past 'Z'
        if self.curr > 25:
            raise IndexError("GraphKeyIterator Overflow")
        # Return the current letter and increment the ASCII value
        letter = chr(self.currSymbol)
        self.curr += 1
        self.currSymbol += 1
        return letter



class CNNgraph(nx.Graph):
    #layers must be different sizes
    def __init__(self):
        super(CNNgraph, self).__init__()
        self.layerID = GraphKeyIterator()
        self._dynamic_attrs = {}

    def addNodeFromMatrix(self, Matrix, val):

        self._dynamic_attrs[Matrix] = val
        pass
    
    def initialiseLayer(self, layerSize:int, name:str=None):
        
        cutoff = 64000
        if layerSize <= 0 or layerSize > cutoff:
            ValueError(f"layersize {layerSize} must be greater than 0 and smaller than {cutoff}")
        
        tempLabel = next(self.layerID)
        nLayer = self.layerID.getIter()

        if isinstance(name, None):
            name = tempLabel

        for i in range(1, layerSize+1):
            self.add_node(f"{name}{i}", layer=nLayer)

    def resolveMatrices(self, *args):

        dims = set()
        for matrix in args:
            if not isinstance(matrix, np.ndarray) or isinstance(matrix, torch.Tensor):
                TypeError(f"matrix is not of type numpy array or torch tensor: {type(matrix)}")
            
            dimensions = matrix.shape
            if len(dimensions) > 2 or len(dimensions) <= 1:
                ValueError(f"matrix is too big or small: {dimensions}")

            if dimensions[0] == dimensions[1]:
                ValueError(f"can't resolve matrix of same dimension size: {dimensions}; please use initialise layer")

            for dim in dimensions:
                dims.add(dim)

        for dim in dims:
            self.initialiseLayer(dim)
        
        
        
        #get shape and initialise layers for all matrices
        
        #check no two dimensions have the same size

        #initialise layers, find layer keys by indexing know layerID

        pass

print("hello")