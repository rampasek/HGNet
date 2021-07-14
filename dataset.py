import json
import os.path as osp
from functools import partial
from typing import NamedTuple

import numpy as np
from numpy.random import default_rng
import torch
import torch_geometric.transforms as T
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from sklearn.model_selection import KFold, StratifiedKFold
from torch_geometric.datasets import Planetoid, TUDataset, QM7b, QM9, ZINC, WikiCS
from torch_geometric.utils import degree

from tasks.islands_dataset import IslandsDataset
from tasks.tree_neighbors_match_dataset import TreeNeighborsMatchDataset

precomputed_max_degree = {
    'COLLAB': 491,
    'IMDB-BINARY': 135,
    'IMDB-MULTI': 88,
}


class Task(NamedTuple):
    level: str
    type: str
    reasoning: str
    is_multi_task: bool = False
    has_predefined_split: bool = False
    model_requirement: str = ""
    evaluator: Evaluator = None

    @property
    def is_node_level(self):
        return self.level == 'node'

    @property
    def is_graph_level(self):
        return self.level == 'graph'

    @property
    def is_classification(self):
        return self.type == 'classification'

    @property
    def is_regression(self):
        return self.type == 'regression'

    @property
    def is_inductive(self):
        return self.reasoning == 'inductive'


def load_dataset(name, opts):
    """
    Load a graph dataset from available PyG datasets.
    :param name: PyG dataset name, e.g. "Cora", "DD", "PROTEINS", "ENZYMES"
    :return:
    """
    if name in ('Cora', 'CiteSeer', 'PubMed', 'CoraSMod', 'CiteSeerSMod', 'PubMedSMod'):
        root = 'Planetoid'
    elif name in ('WikiCS', 'WikiCSSMod'):
        root = 'WikiCS'
    elif name in ['QM7b', 'QM9', 'ZINC']:
        root = name
    elif name.startswith('TNM-depth-'):
        root = 'TreeNeighborsMatch'
    elif name.startswith('Islands-'):
        root = 'Islands'
    elif name.startswith('ogbg-mol'):
        root = 'ogbg'
    else:
        root = 'TUDataset'

    predef_splits = {}
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', root)

    if root == 'Planetoid':
        if name.endswith('SMod'):
            assert opts.sanit_mod_k, "For 'sanitized' modification, set 'sanit_mod_k' arg"
            tf_list = [T.NormalizeFeatures(),
                       partial(sanitize_transductive_task, k=opts.sanit_mod_k, resample=True)]
            dataset = Planetoid(path + f"SModK{opts.sanit_mod_k}", name[:-4],
                                pre_transform=T.Compose(tf_list))
            print(f">> Using sanitized version with k={opts.sanit_mod_k}")
            d = dataset[0]
            for split_name in ['train', 'val', 'test']:
                unq = np.unique(d.y[getattr(d, split_name + '_mask')].detach().numpy(), return_counts=True)
                print(f"     {split_name:5}: {unq}, sum: {unq[1].sum()}")
        else:
            dataset = Planetoid(path, name, split='public',
                                pre_transform=T.NormalizeFeatures())
        # has_predefined_split is actually True, but they are defined through masks
        task = Task(level='node', type='classification',
                    reasoning='transductive', has_predefined_split=False)

    elif root == 'WikiCS':
        def select_predef_split(data, split_idx):
            if data.train_mask.ndimension() == 2:
                data.train_mask = data.train_mask[:, split_idx]
                data.val_mask = data.val_mask[:, split_idx]
                data.stopping_mask = data.stopping_mask[:, split_idx]
            return data

        split_idx = opts.fold if opts.fold >= 0 else 0
        if name.endswith('SMod'):
            assert opts.sanit_mod_k, "For 'sanitized' modification, set 'sanit_mod_k' arg"
            dataset = WikiCS(path + f"SModK{opts.sanit_mod_k}",
                             pre_transform=partial(sanitize_transductive_task, k=opts.sanit_mod_k, resample=True),
                             transform=partial(select_predef_split, split_idx=split_idx))
            print(f">> Using sanitized version with k={opts.sanit_mod_k}")
            d = dataset[0]
            for split_name in ['train', 'val', 'stopping', 'test']:
                unq = np.unique(d.y[getattr(d, split_name + '_mask')].detach().numpy(), return_counts=True)
                print(f"     {split_name:8}: {unq}, sum: {unq[1].sum()}")
        else:
            dataset = WikiCS(path,
                             transform=partial(select_predef_split, split_idx=split_idx))
        # has_predefined_split is actually True, but they are defined through masks
        task = Task(level='node', type='classification',
                    reasoning='transductive', has_predefined_split=False)

    elif root == 'TUDataset':
        # Available datasets: https://chrsmrrs.github.io/datasets/docs/datasets/
        func = None
        if name == "COLLAB" or name.startswith('IMDB'):
            # func = T.Constant()
            func = T.OneHotDegree(max_degree=precomputed_max_degree[name])
        dataset = TUDataset(path, name, pre_transform=func, cleaned=False)
        task = Task(level='graph', type='classification', reasoning='inductive')

    elif root == 'QM7b':
        # molecular graphs in QM7b are fully connected Coulomb Matrices
        func = T.Constant()
        dataset = QM7b(path, pre_transform=func)
        dataset.name = root
        task = Task(level='graph', type='regression', reasoning='inductive')

    elif root == 'QM9':
        # transform = T.Distance(norm=False)  # compute atom distances
        # dataset = QM9(path, transform=transform)
        dataset = QM9(path)
        dataset.name = root
        task = Task(level='graph', type='regression', reasoning='inductive',
                    is_multi_task=True)

    elif root == 'ZINC':
        task = Task(level='graph', type='regression', reasoning='inductive',
                    has_predefined_split=True)
        for split in ('test', 'val', 'train'):
            ds = ZINC(path, subset=True, split=split)
            ds.name = root
            ds.data.x = ds.data.x.float()
            ds.data.y = ds.data.y.unsqueeze(1)
            ds.task = task
            predef_splits[split] = ds
        dataset = predef_splits['train']

    elif root == 'TreeNeighborsMatch':
        # The Tree-NeighborsMatch problem (Alon and Yahav, ICLR2021)
        depth = int(name.split('-')[-1])
        dataset = TreeNeighborsMatchDataset(path, depth=depth)
        task = Task(level='root-node', type='classification', reasoning='inductive',
                    model_requirement='TNM')

    elif root == 'Islands':
        graph_spec = name.split('-')[-1]
        dataset = IslandsDataset(path, graph_spec=graph_spec)
        task = Task(level='graph', type='classification', reasoning='inductive',
                    model_requirement='Islands')

    elif root == 'ogbg':
        task = Task(level='graph', type='classification', reasoning='inductive',
                    is_multi_task=True, has_predefined_split=True,
                    model_requirement='ogbg-mol', evaluator=Evaluator(name))
        dataset = PygGraphPropPredDataset(root=path, name=name)
        dataset.data.y = dataset.data.y.to(torch.float32)
        dataset.task = task
        # if args.feature == 'full':
        #     pass
        # elif args.feature == 'simple':
        #     print('using simple feature')
        #     # only retain the top two node/edge features
        #     dataset.data.x = dataset.data.x[:, :2]
        #     dataset.data.edge_attr = dataset.data.edge_attr[:, :2]

        print(task.evaluator.expected_input_format)
        print(task.evaluator.expected_output_format)

        split_idx = dataset.get_idx_split()
        predef_splits['test'] = dataset[split_idx['test']]
        predef_splits['val'] = dataset[split_idx['valid']]
        predef_splits['train'] = dataset[split_idx['train']]
        dataset = predef_splits['train']

    else:
        assert False, f"Unexpected PyG dataset root: {root}"
    dataset.task = task


    print(f"[*] Loaded dataset '{name}' from '{root}':")
    print(f"  {dataset.data}")
    if task.has_predefined_split:
        print(f"    train: {predef_splits['train']}")
        print(f"    val:   {predef_splits['val']}")
        print(f"    test:  {predef_splits['test']}")
    print(f"  task: {task}")
    print(f"  undirected: {dataset[0].is_undirected()}")
    print(f"  num graphs: {len(dataset)}")
    print(f"  avg num_nodes/graph: {int(dataset.data.x.size(0) / len(dataset))}")
    print(f"  num node features: {dataset.num_node_features}")
    print(f"  num edge features: {dataset.num_edge_features}")
    if hasattr(dataset, 'num_tasks'): print(f"  num tasks: {dataset.num_tasks}")
    print(f"  num classes: {dataset.num_classes}")
    # print(f"  y dim: {dataset.data.y.shape[1]}")

    max_degree = 0
    for i in range(len(dataset)):
        try:
            max_degree = max(degree(dataset[i].edge_index[0]).max().int().item(),
                             max_degree)
        except Exception as e:
            # print(e)
            # print(dataset[i])
            pass
    print(f"  max node degree: {max_degree}")

    # import pprint
    # pprint.pprint(vars(dataset), depth=1)
    print("")

    return dataset, predef_splits, task


def create_crossvalidation_splits(dataset, file_name, k):
    n_samples = len(dataset)
    if dataset.task.is_classification:
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=123)
        kf_split = kf.split(np.zeros(n_samples), dataset.data.y)
    else:
        kf = KFold(n_splits=k, shuffle=True, random_state=123)
        kf_split = kf.split(np.zeros(n_samples))
    splits = {'n_samples': n_samples,
              'n_splits': k,
              'cross_validator': kf.__str__(),
              'dataset': dataset.name
              }
    for i, (_, ids) in enumerate(kf_split):
        splits[i] = ids.tolist()
    with open(file_name, 'w') as f:
        json.dump(splits, f)
    print(f"[*] Saved newly generated CV splits by {kf} to\n{file_name}\n")


def split_dataset(dataset, fold):
    file_name = osp.join(osp.dirname(osp.realpath(__file__)), 'splits', f"{dataset.name}.json")
    if not osp.isfile(file_name):
        create_crossvalidation_splits(dataset, file_name, 10)
    with open(file_name) as f:
        splits = json.load(f)
    assert splits['dataset'] == dataset.name, "Unexpected dataset CV splits"
    assert splits['n_samples'] == len(dataset), "Dataset length does not match"
    assert splits['n_splits'] > fold, "Fold selection out of range"
    k = splits['n_splits']

    test_ids = splits[str(fold)]
    val_ids = splits[str((fold + 1) % k)]
    train_ids = []
    for i in range(k):
        if i != fold and i != (fold + 1) % k:
            train_ids.extend(splits[str(i)])

    return dataset[test_ids], dataset[val_ids], dataset[train_ids]


def sanitize_transductive_task(data, k=1, resample=True, num_train_per_class=20,
                               num_val=500, num_test=1000, seed=321):
    """
    Greedily find a k-hop independent set of the graph nodes (i.e. all selected
    nodes are at least k+1 hops apart) and intersect it with all split masks
    of the @data i.e. data.{train/test/val}_mask.

    data: Single pytorch geometric graph data object
    k (int, optional): Size of the "buffer" between selected nodes
    resample (bool): Resample new dataset split masks
    num_train_per_class (int, optional): The number of training samples per class
    num_val (int, optional): The number of validation samples
    num_test (int, optional): The number of test samples
    seed (int, optional): Random seed
    :return: modified pytorch geometric graph data object
    """
    if k == 0:
        return data

    N = data.num_nodes
    A = [set() for _ in range(N)]
    for u, v in data.edge_index.t().tolist():
        A[u].add(v)
        A[v].add(u)

    for _ in range(k - 1):
        newA = [set() for _ in range(N)]
        for u in range(N):
            for v in A[u]:
                newA[u].update(A[v])
            newA[u].difference_update([u])  # remove self-loop
        A = newA

    rng = default_rng(seed=seed)
    nodes = [rng.choice(N)]  # list of initial "seed" nodes
    neighbors = set.union(*[A[v] for v in nodes])
    assert not set.intersection(neighbors, nodes), f"{nodes} is not a {k}-hop independent set of G"

    indep_nodes = list(nodes)
    available_nodes = set(range(N)).difference(neighbors.union(nodes))
    while available_nodes:
        node = rng.choice(list(available_nodes))
        indep_nodes.append(node)
        available_nodes.difference_update(list(A[node]) + [node])

    print(f"Found {k}-hop Independent Set of size {len(indep_nodes)}")
    indep_nodes = np.asarray(indep_nodes)

    train_nodes_before = data.train_mask.numpy().astype(int).sum()
    val_nodes_before = data.val_mask.numpy().astype(int).sum()
    test_nodes_before = data.test_mask.numpy().astype(int).sum()

    rm_mask = data.train_mask.new_empty(data.train_mask.size(0), dtype=torch.bool)
    rm_mask.fill_(True)
    rm_mask[indep_nodes] = False

    if resample:
        ys = data.y.clone().detach()
        ys[rm_mask] = -1  # don't pick masked-out nodes
        num_classes = ys.max().item() + 1

        if data.train_mask.ndimension() > 1:  # handling WikiCS dataset
            # supporting only a single data split
            data.train_mask = data.train_mask[:, 0]
            data.val_mask = data.val_mask[:, 0]
            data.stopping_mask = data.stopping_mask[:, 0]

        data.train_mask.fill_(False)
        for c in range(num_classes):
            idx = (ys == c).nonzero(as_tuple=False).view(-1)
            idx = idx[rng.permutation(idx.size(0))[:num_train_per_class]]
            data.train_mask[idx] = True

        used = data.train_mask.clone().detach()
        used[rm_mask] = True
        remaining = (~used).nonzero(as_tuple=False).view(-1)
        remaining = remaining[rng.permutation(remaining.size(0))]
        num_remaining = remaining.size(0)

        num_needed = num_val + num_test + (num_val if hasattr(data, 'stopping_mask') else 0)
        print(f"> remaining: {num_remaining}, needed: {num_needed}")
        if num_needed > num_remaining:
            if hasattr(data, 'stopping_mask'):
                num_val = int(num_remaining * 0.25)
                num_test = int(num_remaining * 0.5)
            else:
                num_val = int(num_remaining * 0.333)
                num_test = int(num_remaining * 0.666)
            print(f"> new num_val {num_val}, num_test: {num_test}")

        data.val_mask.fill_(False)
        data.val_mask[remaining[:num_val]] = True
        num_prev = num_val

        if hasattr(data, 'stopping_mask'):
            stop_nodes_before = data.stopping_mask.numpy().astype(int).sum()
            data.stopping_mask.fill_(False)
            data.stopping_mask[remaining[num_prev:num_prev + num_val]] = True
            num_prev += num_val

        data.test_mask.fill_(False)
        data.test_mask[remaining[num_prev:num_prev + num_test]] = True

    else:
        data.train_mask[rm_mask] = False
        data.val_mask[rm_mask] = False
        data.test_mask[rm_mask] = False
        if hasattr(data, 'stopping_mask'):
            stop_nodes_before = data.stopping_mask.numpy().astype(int).sum()
            data.stopping_mask[rm_mask] = False

    train_nodes_after = data.train_mask.numpy().astype(int).sum()
    val_nodes_after = data.val_mask.numpy().astype(int).sum()
    test_nodes_after = data.test_mask.numpy().astype(int).sum()

    print(f">> Sanitizing... found Independent Set of size {len(indep_nodes)}")
    print(f"     train_nodes: before={train_nodes_before}, after={train_nodes_after}")
    print(f"     val_nodes:  before={val_nodes_before}, after={val_nodes_after}")
    if hasattr(data, 'stopping_mask'):
        stop_nodes_after = data.stopping_mask.numpy().astype(int).sum()
        print(f"     stop_nodes:  before={stop_nodes_before}, after={stop_nodes_after}")
    print(f"     test_nodes: before={test_nodes_before}, after={test_nodes_after}")
    print("     all y: ", np.unique(data.y.detach().numpy(), return_counts=True))

    return data