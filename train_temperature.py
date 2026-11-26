#Here we initializate the train as in train.py, but for 3 different
#temperatures for the pooling 'main', which has lowest val error
import json
import os
import hydra
import torch

from src.data_utils.corpus import Corpus
from src.retrievers.dpr import DPR
from src.training.trainer import build_pairs, Trainer

@hydra.main(config_path='config', config_name='fiqa_experiment.yaml', version_base=None)
def main(cfg):
    #Load the data
    corpus = Corpus(dataset_name=cfg.dataset.name, base_dir=f'{cfg.base_path}/retrieval_data')
    train_pairs = build_pairs(corpus.train)
    val_pairs = build_pairs(corpus.val)
    if not torch.cuda.is_available():
        raise RuntimeError('No GPU available. Stopped. ')

    pooling = 'mean'
    cfg.model.pooling_type = pooling

    #Train the same model with different temperatures
    temperatures = [0.02, 0.05, 0.1]
    for temperature in temperatures:
        torch.manual_seed(33)
        print(f'Pooling: {pooling} | Temperature: {temperature} ', flush=True)
        output_direction = f'outputs/dpr_weights/temperature/{pooling}_T{temperature}'
        os.makedirs(output_direction, exist_ok=True)

        dpr = DPR(cfg, corpus.all_documents)
        trainer = Trainer(dpr, lr=1e-4, batch_size=16, epochs=10, temperature=temperature)
        loss = trainer.train(train_pairs, val_dataset=val_pairs)

        # Save the weights of the best epoch
        torch.save(trainer.best_query_weights, f'{output_direction}/query_encoder.pt')
        torch.save(trainer.best_document_weights, f'{output_direction}/document_encoder.pt')
        with open(f'{output_direction}/losses_T{temperature}.json', 'w') as f:
            json.dump({'train_losses': [float(x) for x in trainer.losses],
                    'val_losses': [float(x) for x in trainer.val_losses],
                    'epochs': trainer.epochs,
                    'best_epoch': trainer.best_epoch,
                    'temperature': temperature,
                    'pooling': pooling}, f)
if __name__ == '__main__':
    main()
