from src.retrievers.models.encoder import TransformerEncoder

class DPR:
    def __init__(self, cfg):
        self.cfg = cfg
        self.__load_models()
    
    def __load_models(self):
        # n_heads, n_layers, emb_in, hidden_dim, feed_forward_dim
        query_encoder = TransformerEncoder(n_layers = self.cfg.model.n_layers,
                                           n_heads=self.cfg.model.n_heads,
                                           emb_dim=self.cfg.model.emb_dim,
                                           hidden_dim=self.cfg.model.hidden_dim,
                                           feed_forward_dim=self.cfg.model.feed_forward_dim,
                                           pooling_type=self.cfg.model.pooling_type)

        document_encoder = TransformerEncoder(n_layers = self.cfg.model.n_layers,
                                           n_heads=self.cfg.model.n_heads,
                                           emb_dim=self.cfg.model.emb_dim,
                                           hidden_dim=self.cfg.model.hidden_dim,
                                           feed_forward_dim=self.cfg.model.feed_forward_dim,
                                           pooling_type=self.cfg.model.pooling_type)
        
        query_weights_path = self.cfg.model.get('query_encoder_weights_path', None)
        if self.cfg.model.get('query_encoder_weights_path', None):
            query_encoder.load_state_dict(query_weights_path)

        document_weights_path = self.cfg.model.get('document_encoder_weights_path', None)
        if self.cfg.model.get('query_encoder_weights_path', None):
            document_encoder.load_state_dict(document_weights_path)

    

        

