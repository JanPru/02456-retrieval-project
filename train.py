import json
import os

import torch
import yaml
from transformers import BertTokenizer

from src.data_utils.corpus import Corpus
from src.retrievers.models.encoder import TransformerEncoder
from src.training.trainer import build_pairs, Trainer



#Load the data
corpus = Corpus(dataset_name='fiqa', base_dir='./retrieval_data')
train_pairs = build_pairs(corpus.train)
val_pairs = build_pairs(corpus.val)

#Set up the training
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
with open('config/model/encoder.yaml') as f:
    config = yaml.safe_load(f)
if torch.cuda.is_available(): 
    device = 'cuda'
else: 
    raise RuntimeError('No GPU available. Stopped. ')


#Save losses for the different  poolings
pools = ['cls', 'mean', 'max']
for pooling in pools:
    torch.manual_seed(33)
    print(f'Pooling: {pooling} ', flush=True)
    output_direction = f'outputs/dpr_weights/{pooling}'            
    os.makedirs(output_direction, exist_ok=True)
    query_encoder= TransformerEncoder(n_layers=config['n_layers'],
                              n_heads=config['n_heads'],
                              emb_dim=config['emb_dim'],
                              hidden_dim=config['hidden_dim'],
                              feed_forward_dim=config['feed_forward_dim'],
                              pooling_type=pooling)
    passage_encoder= TransformerEncoder(n_layers=config['n_layers'],
                              n_heads=config['n_heads'],
                              emb_dim=config['emb_dim'],
                              hidden_dim=config['hidden_dim'],
                              feed_forward_dim=config['feed_forward_dim'],
                              pooling_type=pooling)
    trainer=Trainer(query_encoder, passage_encoder, tokenizer, 
                    device, lr=1e-4, batch_size=16, epochs=10, temperature=0.05)
    loss = trainer.train(train_pairs, val_dataset=val_pairs)

    # Save the weights of the best epoch
    torch.save(trainer.best_query_weights, f'{output_direction}/query_encoder.pt')
    torch.save(trainer.best_document_weights, f'{output_direction}/document_encoder.pt')
    with open(f'{output_direction}/losses_{pooling}.json', 'w') as f:
        json.dump({'train_losses': [float(x) for x in trainer.losses],
                'val_losses': [float(x) for x in trainer.val_losses],
                'epochs': trainer.epochs,
                'best_epoch': trainer.best_epoch}, f)

    # ---------- Temporary plot of the losses (to delete later) ----------
    import matplotlib
    matplotlib.use('Agg')   # no screen on the HPC: only save the image
    import matplotlib.pyplot as plt

    steps_per_epoch = len(trainer.losses) // trainer.epochs
    train_per_epoch = [sum(trainer.losses[i*steps_per_epoch:(i+1)*steps_per_epoch]) / steps_per_epoch
                       for i in range(trainer.epochs)]
    epochs = range(1, trainer.epochs + 1)

    plt.figure()
    plt.plot(epochs, train_per_epoch, label='train')
    plt.plot(epochs, trainer.val_losses, label='validation')
    plt.axvline(trainer.best_epoch, linestyle='--', color='gray', label='best epoch')
    plt.title(f'Losses ({pooling} pooling)')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(f'{output_direction}/losses_{pooling}.png')
    plt.close()
    # ---------------------------------------------------------------------