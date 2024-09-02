from dgl.nn.pytorch import EGNNConv, GATConv, GraphormerLayer, SpatialEncoder3d, GCN2Conv, TAGConv
from torch import nn
import torch.nn.functional as F
from torch.nn import LayerNorm, SiLU
from unimol_tools.models import UniMolModel
import torch as th
from utils.data_utils import G2SBatch


class EGNNFeature(nn.Module):

    def __init__(self, args, in_size=82, edge_feat_size=6, hidden_size=256, out_size=256, egnn_layer_num=1,
                 transformer_layer = 0,
                 atten_layers = 0):
        super(EGNNFeature, self).__init__()
        self.args = args
        self.gcn_layer_num = 8
        self.residence = True
        self.new_arch = args.new
        self.egnn = EGNNConv(in_size=in_size,
                             edge_feat_size=edge_feat_size,
                             hidden_size=hidden_size,
                             out_size=out_size
                             )
        # self.layer_norm = LayerNorm(out_size, eps=1e-6)
        self.egnn_layers = nn.ModuleList([EGNNConv(
            in_size=out_size,
            edge_feat_size=0,
            hidden_size=args.encoder_hidden_size,
            out_size=out_size
        ) for i in range(egnn_layer_num - 1)])

        self.gcn_layers = nn.ModuleList(
            [GCN2Conv(
                in_feats=out_size,
                layer= i,
                alpha= 0.5,
                allow_zero_in_degree=True,
                activation= SiLU()
            ) for i in range(1,self.gcn_layer_num+1)]

        )

        # self.gcn_layers = nn.ModuleList(
        #     [
        #         TAGConv(in_feats=out_size,
        #                 out_feats=out_size,
        #                 k=3,
        #                 activation=SiLU()
        #                 ) for i in range(1, self.gcn_layer_num)
        #     ]
        # )
        if self.new_arch:
            self.tf_layers = nn.ModuleList(
                [GraphormerLayer(feat_size=out_size, hidden_size=hidden_size * 2, num_heads= 8, attn_bias_type="add",
                                                 dropout=args.dropout, attn_dropout=args.attn_dropout) for _ in range(transformer_layer)]
            )
            # self.graph_encoder = GraphormerLayer(feat_size=out_size, hidden_size=2048, num_heads=8, attn_bias_type="add",
            #                                      dropout=0.1, attn_dropout=0.1)

            self.spatial_encoder = SpatialEncoder3d(num_kernels=4, num_heads=8, max_node_type=62)
        # self.gnn = WLN(node_in_feats=in_size,
        #                edge_in_feats=edge_feat_size,
        #                node_out_feats=out_size,
        #                n_layers=3)

        # self.gatconv = GATConv(in_feats=out_size, out_feats=out_size, num_heads=args.attn_enc_heads, feat_drop=args.dropout,
        #                        attn_drop=args.attn_dropout)

        # self.embedding = UniMolModel(output_dim=1, data_type='molecule', remove_hs=True).to(self.device)

        # self.gatconv = GAT(in_size=256, hid_size=256,out_size=256, heads=[8, 3], feat_drop=args.dropout,
        #                        attn_drop=args.attn_dropout)

    def forward(self, reaction_batch: G2SBatch):
        graph = reaction_batch.batch_graph

        coordinate = graph.ndata['dist']
        node_feats = graph.ndata['hv']
        edge_features = graph.edata['he']
        # WLN - feature extraction
        # h = self.gnn(graph, node_feats, edge_features)
        h, x = self.egnn(graph, node_feats, coordinate,edge_features)
        residence = h
        # for i, layer in enumerate(self.egnn_layers):
        #     h,_ = layer(graph, h, coordinate)
        # if self.residence:
        #     h = residence + h
        for i, layer in enumerate(self.gcn_layers):
            h = layer(graph, h, residence)


        # h = self.layer_norm(h)
        # return self.gatconv(graph, h).mean(1)

        #(number_nodes, feature_size)
        if self.new_arch:
            max_nodes = th.max(graph.batch_num_nodes())
            sum_num_nodes = 0
            encoder_inputs = []
            encoder_coord = []
            for number_node in graph.batch_num_nodes():
                sub_h = h[sum_num_nodes: sum_num_nodes + number_node]
                sub_coord = coordinate[sum_num_nodes:sum_num_nodes + number_node]
                # n_nodes, feature_size
                m = nn.ZeroPad2d((0, 0, 0, max_nodes - number_node))
                encoder_inputs.append(m(sub_h))
                encoder_coord.append(m(sub_coord))

            inputs = th.stack(encoder_inputs, dim=0)
            coords = th.stack(encoder_coord, dim=0)

            bias = self.spatial_encoder(coords)
            h = inputs
            for layer in self.tf_layers:
                h = layer(h, bias)
        return h

class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, heads, feat_drop, attn_drop):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        # two-layer GAT
        self.gat_layers.append(
            GATConv(
                in_size,
                hid_size,
                heads[0],
                feat_drop=feat_drop,
                attn_drop=attn_drop,
                
                activation=F.elu
            )
        )
        self.gat_layers.append(
            GATConv(
                hid_size * heads[0],
                out_size,
                heads[1],
                feat_drop=feat_drop,
                attn_drop=attn_drop,
                activation=None,
            )
        )

    def forward(self, g, inputs):
        h = inputs
        for i, layer in enumerate(self.gat_layers):
            h = layer(g, h)
            if i == 1:  # last layer
                h = h.mean(1)
            else:  # other layer(s)
                h = h.flatten(1)
        return h