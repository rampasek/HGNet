import os.path as osp

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils.convert import from_networkx

from tasks.islands import IslandDataGenerator, load_euroroad_graph, load_minnesota_graph


class IslandsDataset(InMemoryDataset):
    """
    My Islands dataset.
    """

    def __init__(self, root, graph_spec, transform=None, pre_transform=None):
        self.name = f"Islands-{graph_spec}"
        self.graph_spec = graph_spec
        super().__init__(root, transform, pre_transform)
        self.data, self.slices, self.dim0 = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['dummy_placeholder.txt']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        with open(osp.join(self.raw_dir, 'about.txt'), 'w') as f:
            f.write("No download required for this dataset.\n")
            f.write(f"The {self.name} dataset is generated w/ graph_spec={self.graph_spec}.")

    def process(self):
        generator = IslandDataGenerator(seed=789)
        if self.graph_spec.startswith('Euroroad'):
            G0 = load_euroroad_graph()
            num_nodes = None
        elif self.graph_spec.startswith('Minnesota'):
            G0 = load_minnesota_graph()
            num_nodes = None
        elif self.graph_spec.isnumeric():
            G0 = None
            num_nodes = int(self.graph_spec)
        else:
            assert False, f"Unexpected Islands graph spec: {self.graph_spec}"
        nx_data = generator.generate(num_graphs=7500, num_nodes=num_nodes, G0=G0,
                                     verbose=True)

        dim0 = 2
        data_list = []
        for nxG, label in nx_data:
            data = from_networkx(nxG)
            data.y = label
            data_list.append(data)

        print(f"Generated new dataset with graph_spec={self.graph_spec}")
        print(f"len = {len(data_list)}")
        print(f"dim0 = {dim0}")

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices, dim0), self.processed_paths[0])
