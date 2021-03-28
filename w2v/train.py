from torch.utils.data import DataLoader
from model import model
from trainer.trainer import Trainer, NegSamplingTrainer
import torch.nn as nn
import numpy as np
import argparse
import dataset
import pickle
import torch
import utils
import sys
import os


def main(config):
    print('loading dataset')
    if config.dataset_path == None:
        if config.model == 'cbow':
            nlp_dataset = dataset.cbow_dataset.CBOWDataset(config)
        elif config.model == 'skip-gram':
            nlp_dataset = dataset.skipgram_dataset.SkipGramDataset(config)
        elif config.model == 'neg-sampling':
            nlp_dataset = dataset.negsampling_dataset.NegSamplingDataset(
                config
            )
        elif config.model == 'fast-text':
            nlp_dataset = dataset.fasttext_dataset.FastTextDataset(config)
        else:
            raise AssertionError('dataset should be one of w2v models.')
    else:
        with open(config.dataset_path, 'rb') as f:
            nlp_dataset = pickle.load(f)
    dataloader = DataLoader(
        nlp_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    print('dataloader made')
    if config.model == 'neg-sampling':
        trainer = NegSamplingTrainer(dataloader, config)
    else:
        model = EmbeddingModule(len(nlp_dataset), config).to(config.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        trainer = Trainer(dataloader, model, criterion, optimizer, config)
    print('start training')
    trainer.train()
    model = trainer.model
    with open('./checkpoints/w2v' + config.model + '_model.pkl', 'wb') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='nlp embedding')
    args.add_argument(
        '-m',
        '--model',
        default='neg-sampling',
        type=str,
        help='which model to use',
    )
    args.add_argument(
        '-cp',
        '--config-path',
        default='config.json',
        type=str,
        help='config file path (default: None)',
    )
    args.add_argument(
        '-fp',
        '--file-path',
        default='\\',
        type=str,
        help='path to latest checkpoint (default: None)',
    )
    args.add_argument(
        '-dp',
        '--dataset-path',
        default=None,
        type=str,
        help='if there is a pickled dataset',
    )
    args.add_argument(
        '-d',
        '--device',
        default='cuda:0',
        type=str,
        help='indices of GPUs to enable (default: all)',
    )
    sys.path.append(
        os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
    )
    config = utils.config_parser(args.parse_args())
    main(config)
