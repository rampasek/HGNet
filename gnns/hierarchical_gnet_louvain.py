from functools import partial

import community as community_louvain
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, RGCNConv, avg_pool
from torch_geometric.utils import (add_self_loops, add_remaining_self_loops,
                                   sort_edge_index, remove_self_loops)
from torch_geometric.utils.convert import to_networkx

from gnns.gcn_wparent import GCNConvWParent, GINConvWParent


class HierarchicalGraphNet(torch.nn.Module):
    """The Hierarchical GraphNet
    TODO: update docstring

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output sample.
        depth (int): The depth of the U-Net architecture.
        sum_res (bool, optional): If set to :obj:`False`, will use
            concatenation for integration of skip connections instead
            summation. (default: :obj:`True`)
        act (torch.nn.functional, optional): The nonlinearity to use.
            (default: :obj:`torch.nn.functional.relu`)
    """
    def __init__(self, in_channels, hidden_channels, out_channels, depth,
                 preembed_nodes, no_up_convs, dropout_ratio, normalize,
                 inter_connect='sum', act=F.relu):
        super().__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth
        self.act = act
        self.dropout_ratio = dropout_ratio
        self.normalize = normalize
        assert inter_connect in ('sum', 'concat', 'edge', 'addnode'),\
            f"Unknown inter-layer connection type: {inter_connect}"
        self.inter_connect = inter_connect

        channels = hidden_channels
        norm_class = torch.nn.BatchNorm1d
        # norm_class = torch.nn.LayerNorm

        self.preembed_nodes = preembed_nodes  # Linear embedding of node features
        self.no_up_convs = no_up_convs  # No GNN during hierarchical pooling

        # Convolutions going UP the hierarchy towards coarsest level
        self.node_preembedder = torch.nn.Linear(in_channels, channels)
        self.up_convs = torch.nn.ModuleList()
        self.up_norms = torch.nn.ModuleList()
        if self.preembed_nodes:
            self.up_convs.append(GCNConv(channels, channels, improved=False))
        else:
            self.up_convs.append(GCNConv(in_channels, channels, improved=False))
        self.up_norms.append(norm_class(channels))
        for _ in range(depth):
            self.up_convs.append(GCNConv(channels, channels, improved=False))
            self.up_norms.append(norm_class(channels))
        if self.no_up_convs:  # wipe
            self.up_convs = torch.nn.ModuleList()

        # Convolutions going back DOWN the hierarchy from coarsest to finest level
        in_channels = 2 * channels if inter_connect == 'concat' else channels
        self.down_convs = torch.nn.ModuleList()
        self.down_norms = torch.nn.ModuleList()
        GNN_down_class = None
        if inter_connect == 'edge':
            GNN_down_class = partial(RGCNConv, num_relations=2, aggr='add')
        elif inter_connect == 'addnode':
            GNN_down_class = GCNConvWParent
        else:
            GNN_down_class = partial(GCNConv, improved=False)
        self.down_convs.append(GNN_down_class(in_channels, out_channels))
        self.down_norms.append(norm_class(out_channels))
        for _ in range(depth - 1):
            self.down_convs.append(GNN_down_class(in_channels, channels))
            self.down_norms.append(norm_class(channels))
        if not self.normalize:  # wipe
            self.up_norms = torch.nn.ModuleList()
            self.down_norms = torch.nn.ModuleList()


    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        # TODO: test the edge weights support
        edge_weight = x.new_ones(edge_index.size(1))

        if self.preembed_nodes:
            x = self.node_preembedder(x)
        if not self.no_up_convs:
            x = self.up_convs[0](x, edge_index, edge_weight)
            if self.normalize: x = self.up_norms[0](x)
            x = self.act(x)
            x = F.dropout(x, self.dropout_ratio, training=self.training)

        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        clusters = []
        data = Batch(batch=batch, x=x, edge_index=edge_index, edge_attr=edge_weight)

        # data = Batch(batch=batch, x=x, edge_index=edge_index, edge_attr=edge_weight).to('cpu')
        # for level in range(1, self.depth + 1):
        #     G = to_networkx(data, to_undirected=True)
        #     dendo = community_louvain.generate_dendrogram(G, random_state=1)
        #     partition = community_louvain.partition_at_level(dendo, 0)
        #     # partition = community_louvain.best_partition(GL)
        #     cluster = torch.tensor(list(partition.values()))
        #     data = avg_pool(cluster, data)
        #     clusters += [cluster.to(x.device)]
        #
        # data = Batch(batch=batch, x=x, edge_index=edge_index, edge_attr=edge_weight)
        # for level in range(1, self.depth + 1):
        #     cluster = clusters[level - 1]

        for level in range(1, self.depth + 1):
            G = to_networkx(data, to_undirected=True)
            dendo = community_louvain.generate_dendrogram(G, random_state=1)
            partition = community_louvain.partition_at_level(dendo, 0)
            # partition = community_louvain.best_partition(GL)
            cluster = torch.tensor(list(partition.values()), device=x.device)

            data = avg_pool(cluster, data)
            x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
            # print(f"postpool [{level}]: {x.shape}")

            if not self.no_up_convs or level == self.depth:
                x = self.up_convs[level](x, edge_index, edge_weight)
                if self.normalize: x = self.up_norms[level](x)
                x = self.act(x)
                x = F.dropout(x, self.dropout_ratio, training=self.training)

            if level < self.depth:
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
            clusters += [cluster]

        for level in reversed(range(self.depth)):
            res = xs[level]
            edge_index = edge_indices[level]
            edge_weight = edge_weights[level]
            cluster = clusters[level]

            unpooled = x[cluster]
            unpooled_num_nodes = unpooled.size(0)
            unpooled_num_edges = edge_index.size(1)

            if self.inter_connect == 'sum':
                x = res + unpooled
            elif self.inter_connect == 'concat':
                x = torch.cat((res, unpooled), dim=-1)
            elif self.inter_connect == 'addnode':
                x = res
                parent = unpooled
            elif self.inter_connect == 'edge':
                # add inter-layer connections
                x = torch.cat([res, x], dim=0)
                inter_edge_index = torch.stack(
                    [torch.arange(unpooled_num_nodes),
                     cluster.cpu() + unpooled_num_nodes]).to(x.device)
                edge_index = torch.cat([edge_index,
                                        inter_edge_index,
                                        inter_edge_index[(1, 0),]],
                                       dim=1)
                edge_type = torch.cat([torch.zeros(unpooled_num_edges, dtype=torch.long),
                                       torch.ones(2 * unpooled_num_nodes, dtype=torch.long)],
                                      dim=0).to(x.device)
                # edge_index, edge_type = remove_self_loops(edge_index, edge_type)
                edge_index, edge_type = add_remaining_self_loops(
                    edge_index, edge_type, fill_value=0)
                edge_index, edge_type = sort_edge_index(edge_index, edge_type,
                                                        num_nodes=x.size(0))
            else:
                assert False, f"Unexpected layer connect: {self.inter_connect}"

            # x = self.down_convs[level](x, edge_index, edge_weight)
            if self.inter_connect == 'edge':
                x = self.down_convs[level](x, edge_index, edge_type)
                # drop the meta-nodes
                x = x[:unpooled_num_nodes]
            elif self.inter_connect == 'addnode':
                x = self.down_convs[level](x, parent, edge_index)
            else:
                x = self.down_convs[level](x, edge_index)
            if self.normalize: x = self.down_norms[level](x)
            if level > 0:
                x = self.act(x)
                x = F.dropout(x, self.dropout_ratio, training=self.training)

        return x

    def __repr__(self):
        rep = '{}({}, {}, {}, depth={}, inter_connect={}, preembed_nodes={})'.format(
            self.__class__.__name__, self.in_channels, self.hidden_channels,
            self.out_channels, self.depth, self.inter_connect, self.preembed_nodes)
        rep += '\n'
        rep += super().__repr__()
        return rep
