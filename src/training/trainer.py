import os
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
    def __init__(self, query_encoder: nn.Module, document_encoder: nn.Module, tokenizer, device, lr, batch_size, epochs):
        parameters = list(query_encoder.parameters()) + list(document_encoder.parameters())
        self.device = device
        self.query_encoder = query_encoder.to(device)
        self.document_encoder = document_encoder.to(device)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.losses = []    
        self.val_losses = []  

        self.criterion = DPRLoss()
        self.optimizer = torch.optim.AdamW(parameters, lr=lr)


    def encode(self, encoder, text):
        tokens = self.tokenizer(text, padding=True, truncation=True, max_length=256, return_tensors='pt').to(self.device)
        return encoder(tokens)   

    def train(self, query_encoder: nn.Module, document_encoder: nn.Module, dataset, val_dataset = None, save_dir = None):
        loader = DataLoader(dataset, self.batch_size, shuffle = True)

        self.query_encoder.train()
        self.document_encoder.train()

        for epoch in range(self.epochs): 
            for step, batch in enumerate(loader): 
                queries = list(batch[0])
                passages = list(batch[1])

                self.optimizer.zero_grad()
                q_vec = self.encode(query_encoder, queries)
                d_vec = self.encode(document_encoder, passages)
                loss = self.criterion(q_vec, d_vec)
                loss.backward()
                self.optimizer.step()
                self.losses.append(loss.item())

            print(f'epoch {epoch + 1}/{self.epochs} | loss {loss.item():.4f}')

            if val_dataset is not None:
                val_loader = DataLoader(val_dataset, self.batch_size, shuffle = True, generator = torch.Generator().manual_seed(0))
                self.query_encoder.eval()
                self.document_encoder.eval()
                total = 0
                with torch.no_grad():
                    for batch in val_loader:
                        queries = list(batch[0])
                        passages = list(batch[1])
                        q_vec = self.encode(query_encoder, queries)
                        d_vec = self.encode(document_encoder, passages)
                        total += self.criterion(q_vec, d_vec).item()
        
                self.query_encoder.train()
                self.document_encoder.train()
                val_loss = total / len(val_loader) # mean of all losses
                self.val_losses.append(val_loss)
                print(f'epoch {epoch + 1}/{self.epochs} | val loss {val_loss:.4f}')

        # Fin del entrenamiento (todas las epochs): guardar los pesos finales
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok = True)
            torch.save(self.query_encoder.state_dict(), f'{save_dir}/query_encoder.pt')
            torch.save(self.document_encoder.state_dict(), f'{save_dir}/document_encoder.pt')
            print(f'Final weights stored in {save_dir}')
        return loss