from .transformers import *
from utils.utils_gcn import *
from .gnn_layer import CompGCNConv


class HypRelEncoder(nn.Module):
    """
    Define the encoder
    """
    def __init__(self, graph_repr, config, num_layers, qual=False):
        super(self.__class__, self).__init__()

        self.device = config['DEVICE']
        self.act = torch.tanh if 'ACT' not in config['MODEL'] else config['MODEL']['ACT']

        self.config    = config
        self.n_layer   = num_layers
        self.num_ent   = config['NUM_ENTITIES']
        self.emb_dim   = config['EMBEDDING_DIM']
        self.num_rel   = config['NUM_RELATIONS']
        self.gcn_dim   = config['MODEL']['GCN_DIM']
        self.model_nm  = config['MODEL_NAME'].lower()
        self.fact_encoder_model = config['FACT_ENCODER'] 

        self.ent_skip_matrix = get_param((self.emb_dim * 2, self.emb_dim))
        
        # Storing the KG
        self.edge_index = torch.tensor(graph_repr['edge_index'], dtype=torch.long, device=self.device)
        self.edge_type = torch.tensor(graph_repr['edge_type'], dtype=torch.long, device=self.device)        
        self.quals = torch.tensor(graph_repr['quals'], dtype=torch.long, device=self.device)


        if qual:
            if self.fact_encoder_model == "transformer":
                self.fact_encoder = TransformerTriplets(config)
                self.fact_encoder = self.fact_encoder.to(self.device)
            else:
                self.fact_encoder = nn.Linear(self.emb_dim * 3, self.emb_dim)            
        else:
            self.fact_encoder = None


        # Define Layers
        self.trip_conv1 = CompGCNConv(self.emb_dim, self.gcn_dim, self.num_rel, fact_encoder=self.fact_encoder, act=self.act, params=config)
        self.qual_conv1 = CompGCNConv(self.gcn_dim, self.emb_dim, self.num_rel, fact_encoder=self.fact_encoder, act=self.act, params=config)

        self.trip_conv2 = CompGCNConv(self.emb_dim, self.gcn_dim, self.num_rel, fact_encoder=self.fact_encoder, act=self.act, params=config)
        self.qual_conv2 = CompGCNConv(self.gcn_dim, self.emb_dim, self.num_rel, fact_encoder=self.fact_encoder, act=self.act, params=config)

        self.trip_conv1 = self.trip_conv1.to(self.device)
        self.trip_conv2 = self.trip_conv2.to(self.device)

        self.qual_conv1 = self.qual_conv1.to(self.device)
        self.qual_conv2 = self.qual_conv2.to(self.device)

        # self.drop1 = nn.Dropout(config['MODEL']['ENCODER_DROP_1'])
        # self.drop2 = nn.Dropout(config['MODEL']['ENCODER_DROP_2'])

        self.drop1a = nn.Dropout(config['MODEL']['ENCODER_DROP_1'])
        self.drop2a = nn.Dropout(config['MODEL']['ENCODER_DROP_2'])
        self.drop1b = nn.Dropout(config['MODEL']['ENCODER_DROP_1'])
        self.drop2b = nn.Dropout(config['MODEL']['ENCODER_DROP_2'])



    def forward(self, ent_embs, rel_embs):
        """"
        Pass through encoder

        `aux` params are only used for both encoder

        :return:
        """
        x1, r1 = self.trip_conv1(
                    prop_type="trip",
                    edge_index=self.edge_index, 
                    edge_type=self.edge_type, 
                    x=ent_embs,
                    rel_embed=rel_embs,
                    quals=self.quals
                )
        
        x1 = self.drop1a(x1)

        x1, r1 = self.qual_conv1(
                    prop_type="qual",
                    edge_index=self.edge_index, 
                    edge_type=self.edge_type, 
                    x=x1,
                    rel_embed=r1,
                    quals=self.quals
                )
            

        x1 = self.drop1b(x1)

        x2, r2 = self.trip_conv2(
                    prop_type="trip",
                    edge_index=self.edge_index, 
                    edge_type=self.edge_type, 
                    x=x1,
                    rel_embed=r1,
                    quals=self.quals
                )

        x1 = self.drop2a(x1)

        x2, r2 = self.qual_conv2(
                    prop_type="qual",
                    edge_index=self.edge_index, 
                    edge_type=self.edge_type, 
                    x=x2,
                    rel_embed=r2,
                    quals=self.quals
                )
        
        x2 = self.drop2b(x2)

        if self.config['MODEL']['SKIP']:
            concat_ent = torch.cat((x1, x2), dim=1)
            x = torch.matmul(concat_ent, self.ent_skip_matrix)
            return x, r2

        return x2, r2
