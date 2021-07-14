from functools import partial

import torch
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.data import Batch
from torch_geometric.nn.pool.pool import pool_edge

from gnns.edge_pool import EdgePooling
import gnns.ogbmol_conv
from gnns.gcn_wparent import GCNConvWParent


class HierarchicalGraphNet(torch.nn.Module):
    """The Hierarchical GraphNet
    TODO: update docstring
    """

    def __init__(self, in_channels, hidden_channels, out_channels, depth,
                 no_unpool=False, dropout_ratio=0.5, normalize=True,
                 inter_connect='sum', act=F.relu):
        super().__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth
        self.no_unpool = no_unpool
        self.dropout_ratio = dropout_ratio
        self.act = act
        self.normalize = normalize
        assert normalize == True, 'switching normalization not implemented'
        assert inter_connect in ('sum', 'concat', 'addnode'), \
            f"Unknown inter-layer connection type: {inter_connect}"
        self.inter_connect = inter_connect

        assert hidden_channels == out_channels, "For OGB datasets expecting the same dim."
        channels = hidden_channels
        norm_class = torch.nn.BatchNorm1d
        # norm_class = torch.nn.LayerNorm

        # Convolutions going UP the hierarchy towards coarsest level
        self.atom_encoder = AtomEncoder(channels)
        self.up_convs = torch.nn.ModuleList()
        self.up_edge_encs = torch.nn.ModuleList()
        self.up_norms = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()

        # 1st layer is a GCN with BondEncoder
        self.up_convs.append(gnns.ogbmol_conv.GCNConv(channels))
        self.up_edge_encs.append(BondEncoder(channels))
        self.up_norms.append(norm_class(channels))
        for _ in range(depth):
            self.pools.append(EdgePooling(channels, dropout=0))
            self.up_convs.append(gnns.ogbmol_conv.GCNConv(channels))
            self.up_edge_encs.append(torch.nn.Linear(channels, channels))
            self.up_norms.append(norm_class(channels))

        # Convolutions going back DOWN the hierarchy from coarsest to finest level
        self.down_convs = torch.nn.ModuleList()
        self.down_norms = torch.nn.ModuleList()
        if inter_connect == 'addnode':
            GCN_class = partial(GCNConvWParent, in_dim=channels)
        else:
            GCN_class = gnns.ogbmol_conv.GCNConv
        self.down_convs.append(GCN_class(emb_dim=channels))

        self.down_norms.append(norm_class(channels))
        for _ in range(depth - 1):
            self.down_convs.append(GCN_class(emb_dim=channels))
            self.down_norms.append(norm_class(channels))
        # TODO: Actually use the separate edge encoders for up-convs?
        #       Now keeping the edge representation from up-conv layers
        # self.down_edge_encs = torch.nn.ModuleList()
        # self.down_edge_encs.append(BondEncoder(channels))
        # for _ in range(depth - 1):
        #     self.down_edge_encs.append(torch.nn.Linear(channels, channels))

        if inter_connect == 'concat':
            self.down_node_encs = torch.nn.ModuleList()
            for i in range(depth):
                self.down_node_encs.append(torch.nn.Linear(2 * channels, channels))

    def forward(self, data):
        # edge_index, _ = remove_self_loops(edge_index)
        # h_list = [self.atom_encoder(x)]

        # x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        data.x = self.atom_encoder(data.x)
        data.edge_attr = self.up_edge_encs[0](data.edge_attr)
        data.x = self.up_convs[0](data.x, data.edge_index, data.edge_attr)
        data.x = self.up_norms[0](data.x)
        data.x = self.act(data.x)
        data.x = F.dropout(data.x, self.dropout_ratio, training=self.training)
        xs = [data.x]
        # edge_indices = [data.edge_index]
        edge_attrs = [data.edge_attr]
        unpool_infos = []

        for level in range(1, self.depth + 1):
            # TODO: implement a version of EdgePool that also uses edge features
            new_x, new_edge_index, new_batch, unpool_info = self.pools[level - 1](
                data.x, data.edge_index, data.batch)
            cluster = unpool_info.cluster

            # need to redo edge pooling operation using the computed clusters
            # because EdgePooling doesn't support pooling with edge attributes
            new_edge_index, new_edge_attr = pool_edge(cluster, data.edge_index, data.edge_attr)
            data = Batch(batch=new_batch, x=new_x, edge_index=new_edge_index, edge_attr=new_edge_attr)


            data.edge_attr = self.up_edge_encs[level](data.edge_attr)
            data.x = self.up_convs[level](data.x, data.edge_index, data.edge_attr)
            data.x = self.up_norms[level](data.x)
            data.x = self.act(data.x)
            data.x = F.dropout(data.x, self.dropout_ratio, training=self.training)

            if level < self.depth:
                xs.append(data.x)
                # edge_indices.append(data.edge_index)
                edge_attrs.append(data.edge_attr)
            unpool_infos.append(unpool_info)

        # for ind, val in enumerate(xs):
        #     print(f"   {ind}: x shape = P{val.shape}")

        if self.no_unpool:
            return data

        x = data.x
        for level in reversed(range(self.depth)):
            res = xs[level]
            # tmp_edge_index = edge_indices[level]
            edge_attr = edge_attrs[level]
            unpool_info = unpool_infos[level]

            # invert clustering i.e. "unpool"
            unpooled, edge_index, batch = self.pools[level].unpool(x, unpool_info)
            # assert torch.all(tmp_edge_index == edge_index), 'unexpected!'

            if self.inter_connect == 'sum':
                x = res + unpooled
            elif self.inter_connect == 'concat':
                x = torch.cat((res, unpooled), dim=-1)
                x = self.down_node_encs[level](x)
            elif self.inter_connect == 'addnode':
                x = res
                parent = unpooled
            else:
                assert False, f"Unexpected layer connect: {self.inter_connect}"

            if self.inter_connect == 'addnode':
                x = self.down_convs[level](x, parent, edge_index, edge_attr)
            else:
                x = self.down_convs[level](x, edge_index, edge_attr)
            x = self.down_norms[level](x)
            x = self.act(x)  # if level > 0 else x
            x = F.dropout(x, self.dropout_ratio, training=self.training)

        return x

    def __repr__(self):
        rep = '{}({}, {}, {}, depth={}, inter_connect={}, dropout_ratio={})'.format(
            self.__class__.__name__, self.in_channels, self.hidden_channels,
            self.out_channels, self.depth, self.inter_connect, self.dropout_ratio)
        rep += '\n'
        rep += super().__repr__()
        return rep
