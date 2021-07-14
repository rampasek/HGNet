import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class UnionFind:
    def __init__(self, n):
        self.A = [-1] * n

    def find(self, x):
        if self.A[x] < 0:
            return x
        else:
            self.A[x] = self.find(self.A[x])
            return self.A[x]

    def union(self, a, b):
        a = self.find(a)
        b = self.find(b)
        if a == b:
            return
        if self.A[a] < self.A[b]:
            self.A[a] += self.A[b]
            self.A[b] = a
        else:
            self.A[b] += self.A[a]
            self.A[a] = b


class IslandDataGenerator:
    def __init__(self, seed=321):
        self.rng = np.random.default_rng(seed=seed)

    def _make_2d_grid(self, num_nodes):
        n = int(np.sqrt(num_nodes))
        G = nx.generators.lattice.grid_2d_graph(n, n)
        G = nx.convert_node_labels_to_integers(G)

        for v, d in G.nodes(data=True):
            d['x'] = 0
        return G

    def _create_islands(self, G, start_nodes=[], step_limit=None):
        if not start_nodes:
            for u in self.rng.choice(G.nodes(), size=2):
                start_nodes.append(u)

        budget = len(G) // 2
        colored = 0
        queue = list(start_nodes)
        step = 0
        while queue:
            s = queue.pop(0)
            step += 1
            if colored >= budget or (step_limit is not None and step >= step_limit):
                continue
            if G.nodes[s]['x'] == 0:
                G.nodes[s]['x'] = 1
                colored += 1

            rnum = 1 + self.rng.choice(1, replace=False)
            for t in self.rng.choice(G.adj[s], size=rnum):
                queue.append(t)
        return colored == budget

    @ staticmethod
    def _count_islands(G, color):
        # gdim = np.asarray(G.nodes).max(axis=0) + 1  # dims of the 2-D grid
        # to_int = lambda v: v[0] * gdim[1] + v[1]
        dset = UnionFind(len(G))
        for v in G.nodes:
            if G.nodes[v]['x'] == color:
                for u in G.adj[v]:
                    if G.nodes[u]['x'] == color:
                        dset.union(v, u)
        num_islands = 0
        sizes = []
        for v in G.nodes:
            if G.nodes[v]['x'] == color:
                if dset.A[v] < 0:
                    num_islands += 1
                    sizes.append(-dset.A[v])
        return num_islands, sizes

    def generate(self, num_graphs, num_nodes=None, G0=None, verbose=False):
        """
        Generate a set of `num_graphs` Islands colorings of a 2D grid with total
        of `num_nodes` (if set) or of the provided graph `G0` (if set). The two
        options are mutually exclusive.
        """
        assert (num_nodes is not None) ^ (G0 is not None), \
            f"Only one of num_nodes and G0 parameters can be set"
        label_counts = [0] * 2
        tries = 0
        seen = set()
        graph_list = []
        while len(graph_list) < 2 * num_graphs:
            tries += 1
            if G0 is not None:
                G = nx.Graph(G0)
                if G0.name == 'euroroad':
                    a = self.rng.choice(G.nodes())
                    b = (a + self.rng.choice(70) + 30) % len(G)
                else:
                    a, b = self.rng.choice(G.nodes(), 2)
                step_limit = int(np.sqrt(len(G))) * len(G)
                success = self._create_islands(G, start_nodes=[a, b],
                                               step_limit=step_limit)
            else:
                G = self._make_2d_grid(num_nodes)
                success = self._create_islands(G)
            g = tuple(G.nodes('x'))
            if g in seen or not success:
                if verbose:
                    print('duplicate' if g in seen else 'coloring failed')
                continue
            seen.add(g)
            num_islands, _ = self._count_islands(G, color=1)
            y = num_islands - 1
            if label_counts[y] < num_graphs:
                label_counts[y] += 1
                graph_list.append((G, y))
            if verbose and tries % 100 == 0:
                print(f"... {tries} tries: num islands stats = {label_counts}")
        if verbose:
            print(f"Num islands stats: {label_counts}")
            print(f"Generated {len(graph_list)} unique graphs with {tries} tries.")
        return graph_list


def read_graph_from_file(filename):
    num_nodes, num_edges = None, None
    edges = []
    with open(filename, 'r') as f:
        for line in f:
            if num_nodes is None:  # the first line of the file
                num_nodes, num_edges = [int(s) for s in line.split()]
            else:
                edges.append([int(s) - 1 for s in line.split()])
    assert num_edges == len(edges), \
        f"Read {len(edges)} instead of expected {num_edges} number of edges"

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edges)
    for v in G.nodes:
        G.nodes[v]['x'] = 0
    return G


def load_euroroad_graph():
    import os.path as osp
    fpath = osp.join(osp.dirname(osp.realpath(__file__)), 'euroroad.edges')
    G = read_graph_from_file(fpath)
    G.name = 'euroroad'
    return G


def load_minnesota_graph():
    import os.path as osp
    fpath = osp.join(osp.dirname(osp.realpath(__file__)), 'minnesota.edges')
    G = read_graph_from_file(fpath)
    G.add_edge(348, 354)  # missing edge needed to connect graph
    G.name = 'minnesota'
    return G

def comp_islands_stats(ds, color=1):
    '''
    Plot histograms of how many uncolored nodes have how many colored neighbors
    '''
    stats = [[], []]
    for G, y in ds:
        counts = []
        for v in G.nodes:
            if G.nodes[v]['x'] == color:
                continue
            num_colored_neighbors = 0
            for u in G.adj[v]:
                if G.nodes[u]['x'] == color:
                    num_colored_neighbors += 1
            # if num_colored_neighbors:
            counts.append(num_colored_neighbors)
        stats[y].append(counts)

    fig = plt.figure(figsize=(8, 20))
    mi, mj = 2, 8
    for i in range(mi):
        for j in range(mj):
            ax = fig.add_subplot(mj, mi, j * mi + i + 1)
            ax.hist(stats[i][j], density=False, color=['blue', 'red'][i])
            ax.set_ylabel('Count')
            ax.set_xlabel('# colored neighbors')
    plt.show()


def report_graph_stats(G):
    print(f"Num. nodes:\t{G.number_of_nodes()}")
    print(f"Num. edges:\t{G.number_of_edges()}")
    if nx.is_connected(G):
        print(f"Diameter:\t{nx.algorithms.distance_measures.diameter(G)}")
    else:
        print(f"Diameter:\tG is disconnected; Num. components: {nx.algorithms.components.number_connected_components(G)}")


def test_run_2Dgrid():
    generator = IslandDataGenerator(seed=789)
    ds = generator.generate(num_graphs=10, num_nodes=1024, verbose=True)
    report_graph_stats(ds[0][0])
    # Plot histogram of neighbors' color distribution
    # comp_islands_stats(ds); exit(0)

    fig = plt.figure(figsize=(12, 12))
    for i, gind in enumerate([8, 3, -5, -3]):
        G, y = ds[gind]
        ax = fig.add_subplot(2, 2, i + 1)

        print(f"...plotting G with label=={y}")
        pos = nx.kamada_kawai_layout(G)
        values = [d['x'] for u, d in G.nodes(data=True)]
        nx.draw(G, pos, cmap=plt.get_cmap('coolwarm'), node_color=values,
                with_labels=False, font_color='white', font_size=12, node_size=25,
                ax=ax)
    plt.show()

    print(f"...testing conversion to pytorch_geometric:")
    from torch_geometric.utils.convert import from_networkx
    data = from_networkx(G)
    data.y = y
    print(data)
    print(data.x)


def test_run_euroroad():
    G = load_euroroad_graph()
    report_graph_stats(G)
    print("Original graph components:", end=' ')
    print([len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)])
    # Subset graph to the largest connected component
    # G = G.subgraph(max(nx.connected_components(G), key=len))
    # G = nx.relabel.convert_node_labels_to_integers(G, first_label=0, ordering='default')
    # Use my UnionFind to double-check the subset component
    num_components, components_size = IslandDataGenerator._count_islands(G, 0)
    print(f"Final: # components: {num_components}, their size: {components_size}")
    print(f"Graph is planar: {nx.check_planarity(G)}")

    ## Exploration plotting
    # pos = nx.kamada_kawai_layout(G)
    # # pos = nx.spectral_layout(G)
    # values = [d['x'] for u, d in G.nodes(data=True)]
    # nx.draw(G, pos, cmap=plt.get_cmap('coolwarm'), node_color=values,
    #         with_labels=False, font_color='white', font_size=12, node_size=25)
    # plt.show()
    # exit(0)

    generator = IslandDataGenerator(seed=789)
    ds = generator.generate(num_graphs=10, G0=G, verbose=True)

    fig = plt.figure(figsize=(12, 12))
    for i, gind in enumerate([1, 3, -5, -3]):  # enumerate(range(4)):
        G, y = ds[gind]
        ax = fig.add_subplot(2, 2, i + 1)

        print(f"...plotting G with label=={y}")
        pos = nx.kamada_kawai_layout(G)
        values = [d['x'] for u, d in G.nodes(data=True)]
        nx.draw(G, pos, cmap=plt.get_cmap('coolwarm'), node_color=values,
                with_labels=False, font_color='white', font_size=12, node_size=25,
                ax=ax)
    plt.show()


def test_run_minnesota():
    G = load_minnesota_graph()
    report_graph_stats(G)
    # Use my UnionFind to check the subset component
    num_components, components_size = IslandDataGenerator._count_islands(G, 0)
    print(f"Final: # components: {num_components}, their size: {components_size}")
    print(f"Graph is planar: {nx.check_planarity(G)}")

    ## Exploration plotting
    # # pos = nx.kamada_kawai_layout(G)
    # # pos = nx.spring_layout(G)
    # pos = nx.spectral_layout(G)
    # values = [d['x'] for u, d in G.nodes(data=True)]
    # nx.draw(G, pos, cmap=plt.get_cmap('coolwarm'), node_color=values,
    #         with_labels=False, font_color='white', font_size=12, node_size=25)
    # plt.show()
    # exit(0)

    generator = IslandDataGenerator(seed=789)
    ds = generator.generate(num_graphs=10, G0=G, verbose=True)

    fig = plt.figure(figsize=(12, 12))
    pos = nx.spectral_layout(G)
    for i, gind in enumerate([1, 3, -5, -3]):  # enumerate(range(4)):
        G, y = ds[gind]
        ax = fig.add_subplot(2, 2, i + 1)

        print(f"...plotting G with label=={y}")
        values = [d['x'] for u, d in G.nodes(data=True)]
        nx.draw(G, pos, cmap=plt.get_cmap('coolwarm'), node_color=values,
                with_labels=False, font_color='white', font_size=12, node_size=25,
                ax=ax)
    plt.show()

if __name__ == "__main__":
    test_run_2Dgrid()
    # test_run_euroroad()
    # test_run_minnesota()
