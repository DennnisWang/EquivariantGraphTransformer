import torch
from dgl.nn.pytorch import EGNNConv, GATConv
from dgllife.model import WLN
from torch import nn
import torch.nn.functional as F
from torch.nn import Linear
from unimol_tools.models import UniMolModel

from utils.data_utils import G2SBatch


class UniEGNNFeature(nn.Module):

    def __init__(self, args, in_size=512, edge_feat_size=6, hidden_size=256, out_size=256, egnn_layer_num=1,
                 atten_layers=0):
        super(UniEGNNFeature, self).__init__()
        self.args = args
        self.embedding = UniMolModel(output_dim=1, data_type='molecule', remove_hs=True)
        # self.egnn = EGNNConv(in_size=in_size,
        #                      edge_feat_size=edge_feat_size,
        #                      hidden_size=args.encoder_hidden_size,
        #                      out_size=out_size
        #                      )

        self.fc = Linear(in_features=in_size,out_features=out_size)

        # self.gnn = WLN(node_in_feats=in_size,
        #                edge_in_feats=edge_feat_size,
        #                node_out_feats=out_size,
        #                n_layers=3)

        # frozen all layers
        # for param in self.embedding.parameters():
        #     param.requires_grad = False
        #
        # for param in self.embedding.gbf_proj.parameters():
        #     param.requires_grad = True
        #
        # for param in self.embedding.gbf.parameters():
        #     param.requires_grad = True
        #
        # for param in self.embedding.classification_head.parameters():
        #     param.requires_grad = True

        self.egnn_layers = nn.ModuleList([EGNNConv(
            in_size=out_size,
            edge_feat_size=0,
            hidden_size=args.encoder_hidden_size,
            out_size=out_size
        ) for i in range(egnn_layer_num - 1)])

        # self.gatconv = GATConv(in_feats=256, out_feats=256, num_heads=args.attn_enc_heads, feat_drop=args.dropout,
        #                        attn_drop=args.attn_dropout)

        # self.embedding = UniMolModel(output_dim=1, data_type='molecule', remove_hs=True).to(self.device)

        # self.gatconv = GAT(in_size=256, hid_size=256,out_size=256, heads=[8, 3], feat_drop=args.dropout,
        #                        attn_drop=args.attn_dropout)

    def forward(self, reaction_batch: G2SBatch):
        # graph = reaction_batch.batch_graph
        net_input = reaction_batch.batch_input
        embedding = self.embedding(**net_input, return_repr=True,
                                   return_atomic_reprs=True)
        # coordinate = embedding['atomic_coords']
        node_feats = embedding['atomic_reprs']

        # edge_features = graph.edata['he']
        # coordinate = torch.cat(coordinate, dim=0)
        node_feats = torch.cat(node_feats, dim=0)

        return self.fc(node_feats)

        # WLN - feature extraction
        # h = self.gnn(graph, node_feats, edge_features)

        # h, x = self.egnn(graph, node_feats, coordinate, edge_features)
        # for i, layer in enumerate(self.egnn_layers):
        #     h,_ = layer(graph, h, coordinate)
        # return self.gatconv(graph, h).mean(1)
        # return h


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
