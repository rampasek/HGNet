from distutils.util import strtobool
from functools import partial

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GINConv, ChebConv
from torch_geometric.nn import (global_add_pool, global_mean_pool, global_max_pool,
                                global_sort_pool, GlobalAttention, Set2Set)
from torch_geometric.utils import dropout_adj
from torch_geometric.utils.repeat import repeat

import gnns.hierarchical_gnet_edgepool as HGNet_edgepool
import gnns.hierarchical_gnet_louvain as HGNet_louvain
import gnns.hierarchical_gnet_ogbgmol_edgepool as HGNet_ogbgmol_edgepool
import gnns.hierarchical_gnet_ogbgmol_louvain as HGNet_ogbgmol_louvain
import gnns.ogbmol_conv
from gnns.graph_unet import GraphUNet


# from torch_geometric.nn import GraphUNet  # PyG UNet

def parse_list(string, dtype=int):
    return [dtype(x) for x in string.split(',') if x.strip()]


def parse_activation(activation_name):
    return {
        'ELU': F.elu,
        'LeakyReLU': F.leaky_relu,
        'ReLU': F.relu,
        'Tanh': F.tanh
    }.get(activation_name, None)


class BaseGraphNetModel(torch.nn.Module):
    def __init__(self, dataset, task, opts):
        super().__init__()
        assert task.level in ['node', 'root-node', 'graph'], \
            f"Unexpected task level: {task.level}"
        self.task = task

        ### Input/Output dimensions
        if task.level in ['node', 'graph']:
            self.in_dim = dataset.num_features
        else:
            self.in_dim = opts.hidden_dim
            # GAT is a special case
            if hasattr(opts, 'heads'):
                first_layer_heads = parse_list(opts.heads, dtype=int)[0]
                self.in_dim = opts.hidden_dim * first_layer_heads

        if task.is_multi_task:  # multi-task with one output per task
            pred_dim = dataset.data.y.shape[1]
        else:  # single-task multi-class
            pred_dim = dataset.num_classes

        if task.level == 'node':
            self.out_dim = pred_dim
        else:
            self.out_dim = opts.hidden_dim
            # GAT is a special case, since the attention heads get concatenated
            if hasattr(opts, 'heads'):
                last_layer_heads = parse_list(opts.heads, dtype=int)[-1]
                self.out_dim = opts.hidden_dim * last_layer_heads

        ### Task prediction heads
        if task.level == 'graph':
            # Graph-level classification task
            if opts.global_pooling == 'sort':
                if opts.sort_pool_k < 1:
                    graph_sizes = sorted([dataset[i].x.size(0) for i in range(len(dataset))])
                    percentile_index = int(len(graph_sizes) * opts.sort_pool_k)
                    opts.sort_pool_k = graph_sizes[percentile_index]
                opts.sort_pool_k = int(opts.sort_pool_k)
                opts.sort_pool_k = opts.sort_pool_k
                print(f"Graph classifier: sort_pool_k = {opts.sort_pool_k}")
                self.classifier_in_dim = opts.sort_pool_k * self.out_dim
                self.global_pool = partial(global_sort_pool, k=opts.sort_pool_k)
            elif opts.global_pooling == 'sum':
                self.classifier_in_dim = self.out_dim
                self.global_pool = global_add_pool
            elif opts.global_pooling == 'mean':
                self.classifier_in_dim = self.out_dim
                self.global_pool = global_mean_pool
            elif opts.global_pooling == 'max':
                self.classifier_in_dim = self.out_dim
                self.global_pool = global_max_pool
            elif opts.global_pooling == 'attention':
                self.classifier_in_dim = self.out_dim
                self.global_pool = GlobalAttention(
                    gate_nn=torch.nn.Sequential(torch.nn.Linear(self.out_dim, 2 * self.out_dim),
                                                torch.nn.BatchNorm1d(2 * self.out_dim),
                                                torch.nn.ReLU(),
                                                torch.nn.Linear(2 * self.out_dim, 1)))
            elif opts.global_pooling == 'set2set':
                self.classifier_in_dim = self.out_dim
                self.global_pool = Set2Set(self.out_dim, processing_steps=2)
            else:
                assert False, f"Unexpected global pooling type: {opts.global_pooling}"

            self.graph_predictor = torch.nn.Sequential(
                torch.nn.Linear(self.classifier_in_dim, opts.classifier_dim),
                # torch.nn.LayerNorm(self.classifier_in_dim),
                getattr(torch.nn, opts.activation_classifier)(),
                torch.nn.Dropout(opts.dropout_classifier) if opts.dropout_classifier > 0
                                                          else torch.nn.Identity(),
                torch.nn.Linear(opts.classifier_dim, pred_dim)
            )
        elif task.level == 'root-node':
            # Root node (single node) classification task
            self.root_predictor = torch.nn.Linear(
                in_features=self.out_dim, out_features=pred_dim, bias=True)
        else:
            # Individual node representation / classification task
            pass  # nothing to do

        ### Special dataset-model requirements / feature embeddings
        if task.model_requirement == 'TNM':
            self.layer0_keys = torch.nn.Embedding(
                num_embeddings=dataset.dim0 + 1,
                embedding_dim=self.in_dim)
            self.layer0_values = torch.nn.Embedding(
                num_embeddings=dataset.dim0 + 1,
                embedding_dim=self.in_dim)
        elif task.model_requirement == 'Islands':
            self.color_embedding = None
            # self.in_dim = opts.hidden_dim
            # self.color_embedding = torch.nn.Embedding(
            #     num_embeddings=dataset.dim0,
            #     embedding_dim=self.in_dim)
        elif task.model_requirement == 'ogbg-mol':
            # self.node_encoder = AtomEncoder(opts.hidden_dim)
            # self.edge_encoder = BondEncoder(opts.hidden_dim)
            pass

        self.opts = opts

    @classmethod
    def add_config_arguments(cls, parser):
        parser.add_argument('--hidden_dim', type=int, default=32,
                            help='Node embedding size in graph net')

        parser.add_argument('--global_pooling', type=str, default='sort',
                            help='Type of pooling to create graph representation from node' + \
                                 'embeddings before applying a graph classifier')
        parser.add_argument('--sort_pool_k', type=float, default=30,
                            help='Number of nodes or ratio for sort_pool layer in graph classifier')
        parser.add_argument('--classifier_dim', type=int, default=200,
                            help='Hidden layer size in the graph classifier')

        parser.add_argument('--dropout_attributes', type=float, default=0,
                            help='Dropout rate for node attributes')
        parser.add_argument('--dropout_adjacency', type=float, default=0,
                            help='Dropout rate for graph the adjacency matrix')
        parser.add_argument('--dropout_classifier', type=float, default=0,
                            help='Dropout rate in the classifier')
        parser.add_argument('--activation_network', type=str, default='LeakyReLU')
        parser.add_argument('--activation_classifier', type=str, default='ELU')

        parser.add_argument('--sanit_mod_k', type=int, default=0,
                            help='Size of k-hop neighborhood to sanitize around each labeled node')
        return parser


class HierarchicalGraphNetModel(BaseGraphNetModel):
    """
    Hierarchical Graph Net, a variation on Graph U-Net
    """
    def __init__(self, dataset, task, opts):
        super().__init__(dataset, task, opts)
        opts = self.opts

        if opts.hierarchical_pooling == 'EdgePooling':
            if task.model_requirement == 'ogbg-mol':
                self.gnet = HGNet_ogbgmol_edgepool.HierarchicalGraphNet(
                    in_channels=self.in_dim,
                    hidden_channels=opts.hidden_dim,
                    out_channels=self.out_dim,
                    depth=opts.depth,
                    no_unpool=opts.no_unpool,
                    dropout_ratio=opts.dropout_attributes,
                    normalize=opts.use_layer_norm,
                    inter_connect=opts.inter_connect,
                    act=parse_activation(opts.activation_network)
                )
            else:
                self.gnet = HGNet_edgepool.HierarchicalGraphNet(
                    in_channels=self.in_dim,
                    hidden_channels=opts.hidden_dim,
                    out_channels=self.out_dim,
                    depth=opts.depth,
                    preembed_nodes=opts.preembed_nodes,
                    no_up_convs=opts.no_up_convs,
                    dropout_ratio=opts.dropout_attributes,
                    normalize=opts.use_layer_norm,
                    inter_connect=opts.inter_connect,
                    act=parse_activation(opts.activation_network)
                )
        elif opts.hierarchical_pooling == 'Louvain':
            if task.model_requirement == 'ogbg-mol':
                self.gnet = HGNet_ogbgmol_louvain.HierarchicalGraphNet(
                    in_channels=self.in_dim,
                    hidden_channels=opts.hidden_dim,
                    out_channels=self.out_dim,
                    depth=opts.depth,
                    no_unpool=opts.no_unpool,
                    dropout_ratio=opts.dropout_attributes,
                    normalize=opts.use_layer_norm,
                    inter_connect=opts.inter_connect,
                    act=parse_activation(opts.activation_network)
                )
            else:
                self.gnet = HGNet_louvain.HierarchicalGraphNet(
                    in_channels=self.in_dim,
                    hidden_channels=opts.hidden_dim,
                    out_channels=self.out_dim,
                    depth=opts.depth,
                    preembed_nodes=opts.preembed_nodes,
                    no_up_convs=opts.no_up_convs,
                    dropout_ratio=opts.dropout_attributes,
                    normalize=opts.use_layer_norm,
                    inter_connect=opts.inter_connect,
                    act=parse_activation(opts.activation_network)
                )
        else:
            assert False, f"Unexpected pooling method {opts.hierarchical_pooling}"

        self.opts = opts

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else None

        if self.task.model_requirement == 'TNM':
            # Learnable embedding layers
            x_key, x_val = x[:, 0], x[:, 1]
            x_key_embed = self.layer0_keys(x_key)
            x_val_embed = self.layer0_values(x_val)
            x = x_key_embed + x_val_embed
        elif self.task.model_requirement == 'Islands':
            if self.color_embedding is not None:
                x = self.color_embedding(x)
            else:
                x = x.unsqueeze(1).float()

        if self.task.model_requirement == 'ogbg-mol':
            out = self.gnet(data)
            if hasattr(out, 'batch'):  # no_unpool==True returns Batch object
                out, batch = out.x, out.batch
        else:
            if self.opts.dropout_adjacency > 0:
                edge_index, _ = dropout_adj(edge_index,
                                            p=self.opts.dropout_adjacency,
                                            force_undirected=True,
                                            training=self.training)
            # if self.opts.dropout_attributes > 0:
            #     x = F.dropout(x, p=self.opts.dropout_attributes, training=self.training)
            out = self.gnet(x, edge_index, batch)

        if self.task.level == 'graph':
            out = self.global_pool(out, batch)
            out = self.graph_predictor(out)
        elif self.task.level == 'root-node':
            root_nodes = out[data.root_mask]
            out = self.root_predictor(root_nodes)
        return out

    @classmethod
    def add_config_arguments(cls, parser):
        parser = super().add_config_arguments(parser)
        parser.add_argument('--depth', type=int, default=3,
                            help='Hierarchy depth')
        parser.add_argument('--hierarchical_pooling', type=str, default='ASAPooling',
                            help='The pooling method to create hierarchical graph representation')
        parser.add_argument('--pool_ratios', type=str, default='2000,0.5',
                            help='Count or ratio of nodes to sample in pooling layers')
        parser.add_argument('--no_unpool', type=lambda x: bool(strtobool(x)), default=False,
                            help='Do not unpool the hierarchy, return the coarsest representation')
        parser.add_argument('--preembed_nodes', type=lambda x: bool(strtobool(x)), default=False,
                            help='Apply a node feature embedding layer first')
        parser.add_argument('--no_up_convs', type=lambda x: bool(strtobool(x)), default=False,
                            help='Do not apply GCN bewteen down-pooling layers going the hierarchy')
        parser.add_argument('--use_layer_norm', type=lambda x: bool(strtobool(x)), default=False,
                            help='Use LayerNorm between GNN layers')
        parser.add_argument('--inter_connect', type=str, default='addnode',
                            help='Adjacent level interconnection type: "sum", "concat", "edge" or "addnode"')
        # parser.add_argument('--normalization', default='none',
        #                     help="Normalization type, 'none' (default) or 'batch'")
        # parser.add_argument('--layers', type=str, default='1,2')
        return parser


class GraphUNetModel(BaseGraphNetModel):
    def __init__(self, dataset, task, opts):
        super().__init__(dataset, task, opts)
        opts = self.opts

        opts.pool_ratios = parse_list(opts.pool_ratios, dtype=float)
        opts.pool_ratios = repeat(opts.pool_ratios, opts.depth)
        avg_num_nodes = [int(dataset.data.x.size(0) / len(dataset))]
        for i in range(len(opts.pool_ratios)):
            if opts.pool_ratios[i] > 1:
                opts.pool_ratios[i] /= avg_num_nodes[i]
            avg_num_nodes.append(avg_num_nodes[i] * opts.pool_ratios[i])
        print(f"GraphUNet pool ratios: {opts.pool_ratios}")
        print(f"GraphUNet avg pooled count: {avg_num_nodes}")

        self.unet = GraphUNet(self.in_dim,
                              opts.hidden_dim,
                              self.out_dim,
                              depth=opts.depth,
                              pool_ratios=opts.pool_ratios,
                              act=parse_activation(opts.activation_network))
        self.opts = opts

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else None

        if self.task.model_requirement == 'TNM':
            # Learnable embedding layers
            x_key, x_val = x[:, 0], x[:, 1]
            x_key_embed = self.layer0_keys(x_key)
            x_val_embed = self.layer0_values(x_val)
            x = x_key_embed + x_val_embed
        elif self.task.model_requirement == 'Islands':
            if self.color_embedding is not None:
                x = self.color_embedding(x)
            else:
                x = x.unsqueeze(1).float()

        if self.opts.dropout_adjacency > 0:
            edge_index, _ = dropout_adj(edge_index,
                                        p=self.opts.dropout_adjacency,
                                        force_undirected=True,
                                        training=self.training)
        if self.opts.dropout_attributes > 0:
            x = F.dropout(x, p=self.opts.dropout_attributes, training=self.training)
        out = self.unet(x, edge_index, batch)

        if self.task.level == 'graph':
            out = self.global_pool(out, batch)
            out = self.graph_predictor(out)
        elif self.task.level == 'root-node':
            root_nodes = out[data.root_mask]
            out = self.root_predictor(root_nodes)
        return out

    @classmethod
    def add_config_arguments(cls, parser):
        parser = super().add_config_arguments(parser)
        parser.add_argument('--depth', type=int, default=3,
                            help='Hierarchy depth')
        parser.add_argument('--pool_ratios', type=str, default='2000,0.5',
                            help='Count or ratio of nodes to sample in gPool layers')
        return parser


class GATConvModel(BaseGraphNetModel):
    def __init__(self, dataset, task, opts):
        super().__init__(dataset, task, opts)
        opts = self.opts

        opts.heads = repeat(parse_list(opts.heads, dtype=int), opts.num_layers)
        opts.heads = opts.heads[:opts.num_layers]
        self.activation = parse_activation(opts.activation_network)

        self.convs = torch.nn.ModuleList()
        self.layer_norms = torch.nn.ModuleList()
        for layer in range(opts.num_layers):
            is_last_layer = bool(layer == opts.num_layers - 1)
            in_channels = self.in_dim if layer == 0 else opts.hidden_dim * opts.heads[layer - 1]
            out_channels = self.out_dim // opts.heads[layer] if is_last_layer else opts.hidden_dim
            self.convs.append(GATConv(in_channels,
                                      out_channels,
                                      heads=opts.heads[layer],
                                      concat=True,
                                      dropout=opts.dropout_attention))
            if opts.use_layer_norm:
                # self.layer_norms.append(torch.nn.LayerNorm(out_channels * opts.heads[layer]))
                self.layer_norms.append(torch.nn.BatchNorm1d(out_channels * opts.heads[layer]))
        self.opts = opts

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else None

        if self.task.model_requirement == 'TNM':
            # Learnable embedding layers
            x_key, x_val = x[:, 0], x[:, 1]
            x_key_embed = self.layer0_keys(x_key)
            x_val_embed = self.layer0_values(x_val)
            x = x_key_embed + x_val_embed
        elif self.task.model_requirement == 'Islands':
            if self.color_embedding is not None:
                x = self.color_embedding(x)
            else:
                x = x.unsqueeze(1).float()

        h_list = [x]
        for layer in range(len(self.convs)):
            h = self.convs[layer](h_list[layer], edge_index)
            if self.opts.use_layer_norm:
                h = self.layer_norms[layer](h)
            if layer != len(self.convs) - 1:
                h = self.activation(h)
                if self.opts.dropout_attributes > 0:
                    h = F.dropout(h, p=self.opts.dropout_attributes, training=self.training)
            if self.opts.use_residual:
                h += h_list[layer]
            h_list.append(h)
        out = h_list[-1]

        if self.task.level == 'graph':
            out = self.global_pool(out, batch)
            out = self.graph_predictor(out)
        elif self.task.level == 'root-node':
            root_nodes = out[data.root_mask]
            out = self.root_predictor(root_nodes)
        return out

    @classmethod
    def add_config_arguments(cls, parser):
        parser = super().add_config_arguments(parser)
        parser.add_argument('--dropout_attention', type=float, default=0.6,
                            help='Dropout of GAT attention coefficients')
        parser.add_argument('--heads', type=str, default='8, 8',
                            help='Number of attention heads in GAT layers; ' + \
                            'if this list is shorter than num_layers then the last is repeated')
        parser.add_argument('--num_layers', type=int, default=2,
                            help='Number of Graph Conv layers')
        parser.add_argument('--use_residual', type=lambda x: bool(strtobool(x)), default=False,
                            help='Add residual connections "x + prev_x"')
        parser.add_argument('--use_layer_norm', type=lambda x: bool(strtobool(x)), default=False,
                            help='Use LayerNorm between GNN layers')
        return parser


class GCNConvModel(BaseGraphNetModel):
    def __init__(self, dataset, task, opts):
        super().__init__(dataset, task, opts)
        opts = self.opts

        self.activation = parse_activation(opts.activation_network)
        is_transductive = not self.task.is_inductive
        self.use_virtualnode = opts.gnn_type.endswith('+vn')
        self.gnn_layer_type = opts.gnn_type.split('+')[0]

        if task.model_requirement == 'ogbg-mol':
            if self.use_virtualnode:
                gnn_class = gnns.ogbmol_conv.GNN_node_Virtualnode
            else:
                gnn_class = gnns.ogbmol_conv.GNN_node
            assert opts.use_layer_norm == True, 'switching normalization not implemented'
            self.gnn_ogbmol = gnn_class(opts.num_layers,
                                        opts.hidden_dim,
                                        JK='last',
                                        drop_ratio=opts.dropout_attributes,
                                        residual=opts.use_residual,
                                        gnn_type=self.gnn_layer_type)
        else:
            self.convs = torch.nn.ModuleList()
            self.layer_norms = torch.nn.ModuleList()
            for layer in range(opts.num_layers):
                in_channels = self.in_dim if layer == 0 else opts.hidden_dim
                out_channels = self.out_dim if layer == opts.num_layers - 1 else opts.hidden_dim
                if self.gnn_layer_type == 'gcn':
                    self.convs.append(GCNConv(in_channels, out_channels, cached=is_transductive))
                elif self.gnn_layer_type == 'gin':
                    mlp = torch.nn.Sequential(torch.nn.Linear(in_channels, out_channels),
                                              torch.nn.BatchNorm1d(out_channels) if opts.use_layer_norm else torch.nn.Identity(),
                                              torch.nn.ReLU(inplace=True),
                                              torch.nn.Linear(out_channels, out_channels))
                    self.convs.append(GINConv(mlp))
                    # self.convs.append(GINConv(mlp, train_eps=True))
                elif self.gnn_layer_type == 'cheb':
                    self.convs.append(ChebConv(in_channels, out_channels, K=2))
                else:
                    raise ValueError(f"Undefined GNN type called {self.gnn_layer_type} parsed from '{opts.gnn_type}'")
                if opts.use_layer_norm:
                    # self.layer_norms.append(torch.nn.LayerNorm(out_channels))
                    self.layer_norms.append(torch.nn.BatchNorm1d(out_channels))

            if self.use_virtualnode:
                # Set the initial virtual node embedding to 0.
                self.virtualnode_embedding = torch.nn.Embedding(1, opts.hidden_dim)
                torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)
                # List of MLPs to transform virtual node at every layer
                self.mlp_virtualnode_list = torch.nn.ModuleList()
                for layer in range(opts.num_layers - 1):
                    self.mlp_virtualnode_list.append(
                        torch.nn.Sequential(torch.nn.Linear(opts.hidden_dim, 2 * opts.hidden_dim),
                                            torch.nn.BatchNorm1d(2 * opts.hidden_dim) if opts.use_layer_norm else torch.nn.Identity(),
                                            torch.nn.ReLU(inplace=True),
                                            torch.nn.Linear(2 * opts.hidden_dim, opts.hidden_dim),
                                            torch.nn.BatchNorm1d(opts.hidden_dim) if opts.use_layer_norm else torch.nn.Identity(),
                                            torch.nn.ReLU(inplace=True)))

        self.opts = opts

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else None

        if self.task.model_requirement == 'TNM':
            # Learnable embedding layers
            x_key, x_val = x[:, 0], x[:, 1]
            x_key_embed = self.layer0_keys(x_key)
            x_val_embed = self.layer0_values(x_val)
            x = x_key_embed + x_val_embed
        elif self.task.model_requirement == 'Islands':
            if self.color_embedding is not None:
                x = self.color_embedding(x)
            else:
                x = x.unsqueeze(1).float()

        if self.task.model_requirement == 'ogbg-mol':
            out = self.gnn_ogbmol(data)
        else:
            num_layers = len(self.convs)
            # virtual node embeddings - one for each graph in the batch
            if self.use_virtualnode:
                if batch is None:
                    batch = edge_index.new_zeros(x.size(0))
                virtualnode_embedding = self.virtualnode_embedding(
                    torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))

            h_list = [x]
            for layer in range(num_layers):
                # if self.use_virtualnode and layer > 0:
                #     # Add message from virtual nodes to graph nodes
                #     h_list[layer] = h_list[layer] + virtualnode_embedding[batch]

                # Message passing among graph nodes
                h = self.convs[layer](h_list[layer], edge_index)

                if self.opts.use_layer_norm:
                    h = self.layer_norms[layer](h)
                if layer != num_layers - 1:
                    h = self.activation(h)
                    if self.opts.dropout_attributes > 0:
                        h = F.dropout(h, p=self.opts.dropout_attributes, training=self.training)
                if self.opts.use_residual:
                    h += h_list[layer]
                h_list.append(h)

                # Update the virtual nodes
                if self.use_virtualnode and layer < num_layers - 1:
                    # add message from graph nodes to virtual nodes
                    vn_temp = global_add_pool(h_list[-1], batch) + virtualnode_embedding
                    # transform virtual nodes using MLP
                    vn_temp = self.mlp_virtualnode_list[layer](vn_temp)
                    if self.opts.dropout_attributes > 0:
                        vn_temp = F.dropout(vn_temp, p=self.opts.dropout_attributes, training=self.training)
                    if self.opts.use_residual:
                        virtualnode_embedding = vn_temp + virtualnode_embedding
                    else:
                        virtualnode_embedding = vn_temp
                    # Add message from virtual nodes to graph nodes
                    h_list[-1] = h_list[-1] + virtualnode_embedding[batch]

            out = h_list[-1]

        if self.task.level == 'graph':
            out = self.global_pool(out, batch)
            out = self.graph_predictor(out)
        elif self.task.level == 'root-node':
            root_nodes = out[data.root_mask]
            out = self.root_predictor(root_nodes)
        return out

    @classmethod
    def add_config_arguments(cls, parser):
        parser = super().add_config_arguments(parser)
        parser.add_argument('--gnn_type', type=str, default='gcn',
                            help='Which GNN layers to use: gcn, gin, cheb, gcn+vn, gin+vn')
        parser.add_argument('--num_layers', type=int, default=2,
                            help='Number of Graph Conv layers')
        parser.add_argument('--use_residual', type=lambda x: bool(strtobool(x)), default=False,
                            help='Add residual connections "x + prev_x"')
        parser.add_argument('--use_layer_norm', type=lambda x: bool(strtobool(x)), default=False,
                            help='Use LayerNorm between GNN layers')
        return parser