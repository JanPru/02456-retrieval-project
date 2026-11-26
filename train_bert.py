import json
import os

import hydra
import torch

from src.data_utils.corpus import Corpus
from src.retrievers.dpr import DPR
from src.retrievers.models.bert_encoder import BertEncoder
from src.training.trainer import build_pairs, Trainer


@hydra.main(config_path='config', config_name='fiqa_experiment.yaml', version_base=None)
def main(cfg):
    #Load the data
    corpus = Corpus(dataset_name=cfg.dataset.name, base_dir=f'{cfg.base_path}/retrieval_data')
    train_pairs = build_pairs(corpus.train)
    val_pairs = build_pairs(corpus.val)

    if not torch.cuda.is_available():
        raise RuntimeError('No GPU available. Stopped. ')

    pools = ['cls', 'mean', 'max']
    for pooling in pools:
        torch.manual_seed(33)
        print(f'Pooling: {pooling} ', flush=True)
        output_direction = f'outputs/dpr_weights/bert_{pooling}'
        os.makedirs(output_direction, exist_ok=True)

        # DPR with the pretrained BERT instead of our encoder
        dpr = DPR(cfg, corpus.all_documents)
        dpr.query_encoder = BertEncoder(pooling).to(dpr.device)
        dpr.document_encoder = BertEncoder(pooling).to(dpr.device)

        trainer = Trainer(dpr, lr=2e-5, batch_size=16, epochs=3, temperature=0.05)
        loss = trainer.train(train_pairs, val_dataset=val_pairs)

        # Save the weights of the best epoch
        torch.save(trainer.best_query_weights, f'{output_direction}/query_encoder.pt')
        torch.save(trainer.best_document_weights, f'{output_direction}/document_encoder.pt')
        with open(f'{output_direction}/losses_bert_{pooling}.json', 'w') as f:
            json.dump({'train_losses': [float(x) for x in trainer.losses],
                    'val_losses': [float(x) for x in trainer.val_losses],
                    'epochs': trainer.epochs,
                    'best_epoch': trainer.best_epoch}, f)

if __name__ == '__main__':
    main()
