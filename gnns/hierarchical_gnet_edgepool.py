from functools import partial

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, RGCNConv
from torch_geometric.utils import add_self_loops, add_remaining_self_loops, \
    remove_self_loops, sort_edge_index

from gnns.edge_pool import EdgePooling
from gnns.gcn_wparent import GCNConvWParent, GINConvWParent


class HierarchicalGraphNet(torch.nn.Module):
    """The Hierarchical GraphNet
    TODO: update docstring
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

        # Pooling and Convolutions going UP the hierarchy towards coarsest level
        self.node_preembedder = torch.nn.Linear(in_channels, channels)
        self.up_convs = torch.nn.ModuleList()
        self.up_norms = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        if self.preembed_nodes:
            self.up_convs.append(GCNConv(channels, channels))
        else:
            self.up_convs.append(GCNConv(in_channels, channels))
        self.up_norms.append(norm_class(channels))
        for _ in range(depth):
            self.pools.append(EdgePooling(channels, dropout=0))
            self.up_convs.append(GCNConv(channels, channels))
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
            GNN_down_class = GCNConv
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
        # TODO: support edge weights (need to augment the pooling)
        # edge_weight = x.new_ones(edge_index.size(1))

        # x = self.up_convs[0](x, edge_index, edge_weight)
        if self.preembed_nodes:
            x = self.node_preembedder(x)
        if not self.no_up_convs:
            x = self.up_convs[0](x, edge_index)
            if self.normalize: x = self.up_norms[0](x)
            x = self.act(x)
            x = F.dropout(x, self.dropout_ratio, training=self.training)

        xs = [x]
        edge_indices = [edge_index]
        # edge_weights = [edge_weight]
        unpool_infos = []

        for level in range(1, self.depth + 1):
            # edge_index, _ = remove_self_loops(edge_index)
            # edge_index, _ = sort_edge_index(edge_index, num_nodes=x.size(0))
            x, edge_index, batch, unpool_info = self.pools[level - 1](
                x, edge_index, batch)
            # print(f"postpool [{level}]: {x.shape}")

            # x = self.up_convs[level](x, edge_index, edge_weight)
            if not self.no_up_convs or level == self.depth:
                x = self.up_convs[level](x, edge_index)
                if self.normalize: x = self.up_norms[level](x)
                x = self.act(x)
                x = F.dropout(x, self.dropout_ratio, training=self.training)

            if level < self.depth:
                xs += [x]
                edge_indices += [edge_index]
                # edge_weights += [edge_weight]
            unpool_infos += [unpool_info]

        for level in reversed(range(self.depth)):
            res = xs[level]
            # edge_index = edge_indices[level]
            # edge_weight = edge_weights[level]
            unpool_info = unpool_infos[level]

            unpooled, edge_index, batch = self.pools[level].unpool(x, unpool_info)
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
                     unpool_info.cluster.cpu() + unpooled_num_nodes]).to(x.device)
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
        rep = '{}({}, {}, {}, depth={}, inter_connect={})'.format(
            self.__class__.__name__, self.in_channels, self.hidden_channels,
            self.out_channels, self.depth, self.inter_connect)
        rep += '\n'
        rep += super().__repr__()
        return rep
