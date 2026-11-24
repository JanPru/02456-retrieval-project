import copy
import torch
import torch.nn as nn
from src.training.loss import DPRLoss
from torch.utils.data import DataLoader

def build_pairs(dataset):
    queries, list_pasages = list(dataset['question']), list(dataset['text'])
    pairs = []
    for queries, list_pasages in zip(queries, list_pasages):
        for pasage in list_pasages:
            pairs.append((queries, pasage))
    return pairs

class Trainer:
    def __init__(self, query_encoder: nn.Module, document_encoder: nn.Module, tokenizer, device, lr, batch_size, epochs, temperature):
        parameters = list(query_encoder.parameters()) + list(document_encoder.parameters())
        self.device = device
        self.query_encoder = query_encoder.to(device)
        self.document_encoder = document_encoder.to(device)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.losses = []    
        self.val_losses = []  

        self.criterion = DPRLoss(temperature)
        self.optimizer = torch.optim.AdamW(parameters, lr=lr)


    def encode(self, encoder, text):
        tokens = self.tokenizer(text, padding=True, truncation=True, max_length=256, return_tensors='pt').to(self.device)
        return encoder(tokens['input_ids'], tokens['attention_mask'])

    def train(self, dataset, val_dataset):
        loader = DataLoader(dataset, self.batch_size, shuffle = True)

        self.query_encoder.train()
        self.document_encoder.train()

        best_val_loss = float('inf')
        self.best_epoch = None

        for epoch in range(self.epochs):
            epoch_losses = []
            for step, batch in enumerate(loader):
                queries = list(batch[0])
                passages = list(batch[1])

                self.optimizer.zero_grad()
                q_vec = self.encode(self.query_encoder, queries)
                d_vec = self.encode(self.document_encoder, passages)
                loss = self.criterion(q_vec, d_vec)
                loss.backward()
                self.optimizer.step()
                self.losses.append(loss.item())
                epoch_losses.append(loss.item())

            train_loss = sum(epoch_losses) / len(epoch_losses) # mean loss of the epoch
            print(f'epoch {epoch + 1}/{self.epochs} | train loss {train_loss:.4f}', flush = True)

            val_loader = DataLoader(val_dataset, self.batch_size, shuffle = True, generator = torch.Generator().manual_seed(0))
            self.query_encoder.eval()
            self.document_encoder.eval()
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    queries = list(batch[0])
                    passages = list(batch[1])
                    q_vec = self.encode(self.query_encoder, queries)
                    d_vec = self.encode(self.document_encoder, passages)
                    total += self.criterion(q_vec, d_vec).item()

            self.query_encoder.train()
            self.document_encoder.train()
            val_loss = total / len(val_loader) 
            self.val_losses.append(val_loss)
            print(f'epoch {epoch + 1}/{self.epochs} | val loss {val_loss:.4f}', flush = True)

            # Weights with the lowest validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_epoch = epoch + 1
                self.best_query_weights = copy.deepcopy(self.query_encoder.state_dict())
                self.best_document_weights = copy.deepcopy(self.document_encoder.state_dict())

        # We load the best weights (lower validation loss) to the final trainer
        self.query_encoder.load_state_dict(self.best_query_weights)
        self.document_encoder.load_state_dict(self.best_document_weights)
        return loss
