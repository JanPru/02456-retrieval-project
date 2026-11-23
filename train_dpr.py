"""
Versión script de train_dpr.ipynb, para lanzarla en el HPC (sin celdas, sin gráficos).
Mismo código que el notebook: cargar FiQA -> parejas -> encoders -> Trainer.
Al final guarda las losses en un JSON para dibujar las curvas luego en el portátil.
"""
import json
import os

import torch
import yaml
from transformers import BertTokenizer

from src.data_utils.corpus import Corpus
from src.retrievers.models.encoder import TransformerEncoder
from src.training.trainer import build_pairs, Trainer

torch.manual_seed(1)

# ---- Datos ----
corpus = Corpus(dataset_name='fiqa', base_dir='./retrieval_data')
pairs = build_pairs(corpus.train)
val_pairs = build_pairs(corpus.val)
print('nº de parejas de train:', len(pairs))
print('nº de parejas de validación:', len(val_pairs))

# ---- Encoders ----
with open('config/model/encoder.yaml') as f:
    cfg_model = yaml.safe_load(f)
print('config del encoder:', cfg_model)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def make_encoder():
    return TransformerEncoder(n_layers=cfg_model['n_layers'],
                              n_heads=cfg_model['n_heads'],
                              emb_dim=cfg_model['emb_dim'],
                              hidden_dim=cfg_model['hidden_dim'],
                              feed_forward_dim=cfg_model['feed_forward_dim'],
                              pooling_type=cfg_model['pooling_type'])

query_encoder = make_encoder()
passage_encoder = make_encoder()

# ---- Entrenamiento ----
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device)
SAVE_DIR = 'outputs/dpr_weights'

# Para una prueba corta, cambia pairs por pairs[:200] y val_pairs por val_pairs[:200]
trainer = Trainer(query_encoder, passage_encoder, tokenizer, device, lr=1e-4, batch_size=16, epochs=10)
loss = trainer.train(pairs,
                     val_dataset=val_pairs,
                     save_dir=SAVE_DIR)
print('loss del último batch:', loss.item())
print('val loss por epoch:', trainer.val_losses)

# ---- Guardar las curvas (se dibujan luego con el notebook) ----
os.makedirs(SAVE_DIR, exist_ok=True)
with open(f'{SAVE_DIR}/losses.json', 'w') as f:
    json.dump({'train_losses': [float(x) for x in trainer.losses],
               'val_losses': [float(x) for x in trainer.val_losses],
               'epochs': trainer.epochs}, f)
print('losses guardadas en', f'{SAVE_DIR}/losses.json')
