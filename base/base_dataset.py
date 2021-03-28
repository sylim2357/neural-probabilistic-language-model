from utils import pre_process_raw_article, mecab_tokenize
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from nltk import sent_tokenize
import pandas as pd
import itertools


class BaseDataset(Dataset):
    def __init__(self, csv_file, root_dir, config):
        super(BaseDataset, self).__init__()
        corpus = self.pre_process()
        self.word_to_idx, self.idx_to_word = self.construct_word_matrix()
        self.x, self.y = self.construct_dataset()

    @abstractmethod
    def pre_process():
        pass

    @abstractmethod
    def construct_word_idx():
        pass

    @abstractmethod
    def construct_dataset():
        pass
