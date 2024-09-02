from dgl.nn.pytorch import EGNNConv, GATConv, GraphormerLayer, SpatialEncoder3d, GCN2Conv, TAGConv, GMMConv,EdgeGATConv
from torch import nn
from torch.nn import SiLU, ELU

from utils.data_utils import G2SBatch

class EGNNFeature(nn.Module):
    #82,6
    def __init__(self, args, in_size=125, edge_feat_size=17, out_size=256):

        super(EGNNFeature, self).__init__()
        self.args = args
        self.gcn_layers_num = 8
        self.egnn = EGNNConv(in_size=in_size,
                             edge_feat_size=edge_feat_size,
                             hidden_size=256,
                             out_size=out_size
                             )
        
        # self.gat = EdgeGATConv(in_feats=in_size,edge_feats=edge_feat_size,out_feats=256,
        #                        num_heads=8,feat_drop=0.3,attn_drop=0.3,activation=ELU())

        self.egnn_layers = nn.ModuleList([EGNNConv(
            in_size=out_size,
            edge_feat_size=0,
            hidden_size=256,
            out_size=out_size
        ) for i in range(args.encoder_num_layers - 1)])

        self.gcn_layers = nn.ModuleList(
            [GCN2Conv(
                in_feats=out_size,
                layer=i,
                alpha=0.5,
                allow_zero_in_degree=True,
                activation=ELU()
            ) for i in range(1, self.gcn_layers_num + 1)]
        )


    def forward(self, reaction_batch: G2SBatch):
        graph = reaction_batch.batch_graph
        coordinate = graph.ndata['dist']
        node_feats = graph.ndata['hv']
        edge_feats = graph.edata['he']
        h, _ = self.egnn(graph, node_feats, coordinate, edge_feats)
        # h = self.gat(graph,node_feats,edge_feats)
        # h = h.mean(1)
        residence = h

        for i, layer in enumerate(self.egnn_layers):
            h, _ = layer(graph, h, coordinate)

        if len(self.gcn_layers) == 0:
            h = h + residence

        for i, layer in enumerate(self.gcn_layers):
            h = layer(graph, h, residence)

        # for i, layer in enumerate(self.gmm_layers):
        #     h = layer(graph, h, edge_feats)

        # h = self.cfcov(graph,h,edge_feats)
        return h
