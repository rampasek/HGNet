import os.path as osp

import torch
from torch_geometric.data import InMemoryDataset

from tasks.dictionary_lookup import DictionaryLookupDataset


class TreeNeighborsMatchDataset(InMemoryDataset):
    """
    The Tree-NeighborsMatch problem (Alon and Yahav, ICLR2021)

    https://arxiv.org/pdf/2006.05205
    https://github.com/tech-srl/bottleneck/
    """

    def __init__(self, root, depth, transform=None, pre_transform=None):
        self.name = f"TNM-depth-{depth}"
        self.depth = depth
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
        with open(osp.join(self.raw_dir, 'dummy_placeholder.txt'), 'w') as f:
            f.write("No download required for this dataset.\n")
            f.write(f"The {self.name} dataset is generated w/ depth={self.depth}.")

    def process(self):
        task = DictionaryLookupDataset(self.depth)
        data_list, _, dim0, out_dim, criterion = task.generate_data(1)
        print(f"Generated new dataset with depth={self.depth}")
        print(f"len = {len(data_list)}")
        print(f"dim0 = {dim0}")
        print(f"out_dim = {out_dim}")
        print(f"criterion = {criterion}")

        for i in range(len(data_list)):
            data_list[i].y -= 1

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices, dim0), self.processed_paths[0])
