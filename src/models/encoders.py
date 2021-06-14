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

        self.n_layer   = num_layers
        self.num_ent   = config['NUM_ENTITIES']
        self.emb_dim   = config['EMBEDDING_DIM']
        self.num_rel   = config['NUM_RELATIONS']
        self.gcn_dim   = config['MODEL']['GCN_DIM']
        self.model_nm  = config['MODEL_NAME'].lower()
        self.fact_encoder_model = config['FACT_ENCODER'] 
        
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
        self.conv1 = CompGCNConv(self.emb_dim, self.gcn_dim, self.num_rel, fact_encoder=self.fact_encoder, act=self.act, params=config)
        self.conv2 = CompGCNConv(self.gcn_dim, self.emb_dim, self.num_rel, fact_encoder=self.fact_encoder, act=self.act, params=config)

        self.drop1 = nn.Dropout(config['MODEL']['ENCODER_DROP_1'])
        self.drop2 = nn.Dropout(config['MODEL']['ENCODER_DROP_2'])

        self.conv1 = self.conv1.to(self.device)
        self.conv2 = self.conv2.to(self.device)



    def forward(self, prop_type, ent_embs, rel_embs):
        """"
        Pass through encoder

        :return:
        """
        x, r = self.conv1(
                    prop_type=prop_type,
                    edge_index=self.edge_index, 
                    edge_type=self.edge_type, 
                    x=ent_embs,
                    rel_embed=rel_embs,
                    quals=self.quals
                )
        x = self.drop1(x)

        if self.n_layer == 2:
            x, r = self.conv2(
                        prop_type=prop_type,
                        edge_index=self.edge_index, 
                        edge_type=self.edge_type, 
                        x=x,
                        rel_embed=r,
                        quals=self.quals
                    )
            x = self.drop2(x) 

        return x, r
