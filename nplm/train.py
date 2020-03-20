from torch.utils.data import DataLoader
from model.dataset import NLPCorpusDataset
from model.model import EmbeddingModule
from trainer.trainer import Trainer
from utils import collate_fn
import torch.nn as nn
import numpy as np
import argparse
import pickle
import torch

def config_parser(args):
    data_path = args.file
    config_path = args.config
    # FILE_PATH = 'D:\\data\\text\\news-articles\\kbanker_articles_subtitles.csv'
    # CONFIG_PATH = 'config.json'
    with open(config_path, 'r') as f:
        config = easydict.EasyDict(json.load(f))
    config['device'] = torch.device(args.device)
    config['data_path'] = data_path
    return config

def main(config):
    nlp_dataset = NLPCorpusDataset(csv_file=config.data_path, root_dir='.')
    dataloader = DataLoader(nlp_dataset, batch_size=config.batch_size, \
                            shuffle=False, num_workers=0, collate_fn=collate_fn)
    model = EmbeddingModule(len(nlp_dataset.word_to_idx),\
                            config.embedding_dim, config.h_dim).to(config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    trainer = Trainer(dataset, model, criterion, optimizer, config)
    trainer.train()
    model = trainer.model
    with open('nplm_model.pkl', 'wb') as f:
        pickle.dump(model, f)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='nlp embedding')
    args.add_argument('-c', '--config', default='config.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-f', '--file', default='D:\\data\\text\\news-articles\\kbanker_articles_subtitles.csv', type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default='cuda:0', type=str,
                      help='indices of GPUs to enable (default: all)')
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    config = config_parser(args.parse_args())
    main()
