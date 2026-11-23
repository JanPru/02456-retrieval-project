import torch
from transformers import BertTokenizer
from src.data_utils.dataset import DocumentDataset
from src.retrievers.models.encoder import TransformerEncoder

class DPR:
    def __init__(self, cfg, documents):
        self.cfg = cfg
        self.index = DocumentDataset(documents)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.__load_models()
        self.query_encoder.to(self.device)
        self.document_encoder.to(self.device)
    
    def __load_models(self):
        # n_heads, n_layers, emb_in, hidden_dim, feed_forward_dim
        self.query_encoder = TransformerEncoder(n_layers = self.cfg.model.n_layers,
                                           n_heads=self.cfg.model.n_heads,
                                           emb_dim=self.cfg.model.emb_dim,
                                           hidden_dim=self.cfg.model.hidden_dim,
                                           feed_forward_dim=self.cfg.model.feed_forward_dim,
                                           pooling_type=self.cfg.model.pooling_type)
        
        self.document_encoder = TransformerEncoder(n_layers = self.cfg.model.n_layers,
                                           n_heads=self.cfg.model.n_heads,
                                           emb_dim=self.cfg.model.emb_dim,
                                           hidden_dim=self.cfg.model.hidden_dim,
                                           feed_forward_dim=self.cfg.model.feed_forward_dim,
                                           pooling_type=self.cfg.model.pooling_type)
        
        query_weights_path = self.cfg.model.get('query_encoder_weights_path', None)
        if self.cfg.model.get('query_encoder_weights_path', None):
            self.query_encoder.load_state_dict(query_weights_path)

        document_weights_path = self.cfg.model.get('document_encoder_weights_path', None)
        if self.cfg.model.get('query_encoder_weights_path', None):
            self.document_encoder.load_state_dict(document_weights_path)
    
    def Lookup(self, queries: list[str], k: int) -> list:
        """
        @param queries: The input text(s) to which relevant passages should be found.
        @param k: The number of relevant passages to retrieve for each input query.
        """
        scores = self.CalculateScores(queries)
        ranked_documents = [[d for _, d in sorted(zip(query_scores, self.index.GetDocuments()), key=lambda pair: pair[0], reverse=True)] for query_scores in scores]
        return [ranked_document[:min(k, len(ranked_documents[0]))] for ranked_document in ranked_documents]

    def CalculateScores(self, queries: list[str]):
        """Calculate scores for a query
        
        Args:
            query (str): The query to calculate scores for
        
        Returns:
            np.array: An array of scores for each document

            num_queries x num_documents where (i, j) = ∑_i TF-IDF(q_i, D_j; D) 
        
        """
        # Convert the queries to a vector
        query_vectors = self.EncodeQueries(queries)
        scores = query_vectors @ self.document_vectors.T
        return scores

    def EncodeDocuments(self, batch_size: int = 64):
        documents = self.index.GetDocuments()
        texts = [d.text for d in documents]
        self.document_encoder.eval()

        encoded_d = []
        with torch.no_grad():  
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                tok = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
                ids = tok["input_ids"].to(self.device)
                vecs = self.document_encoder(ids)
                encoded_d.append(vecs)

        self.document_vectors = torch.cat(encoded_d, dim=0)
        return self.document_vectors

    def EncodeQueries(self, queries: list[str]):
        self.query_encoder.eval()
        with torch.no_grad():
            tok = self.tokenizer(queries, padding=True, truncation=True, return_tensors="pt")
            ids = tok["input_ids"].to(self.device)
            query_vectors = self.query_encoder(ids)
        return query_vectors