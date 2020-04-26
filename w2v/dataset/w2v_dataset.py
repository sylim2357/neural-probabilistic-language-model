from utils import pre_process_raw_article, mecab_tokenize
from torch.utils.data import Dataset
from nltk import sent_tokenize
from abc import abstractmethod
import pandas as pd
import itertools


class W2VDataset(Dataset):
    """W2V Dataset

    Args:
        config (dict): hyperparameters

    Attributes:
        root_dir (str): root
        word_to_idx (dict): word_to_idx mapping
        idx_to_word (dict): idx_to_word mapping
        x (list): train data (5-gram)
        y (list): label

    """

    def __init__(self, config):
        if 'pkl' in config.file_path:
            with open(config.file_path, 'rb') as f:
                corpus = pickle.load(f)[:1000]
        elif config.file_path == 'treebank':
            corpus = treebank.sents()
            if config.model == 'fast-text':
                corpus = self.fast_text_pre_process(corpus)
        else:
            articles = pd.read_csv(config.file_path, encoding='utf-8')['article'].dropna().values

            #pre process
            corpus = self.pre_process(articles)

        #construct word matrix
        self.word_to_idx, self.idx_to_word = self.construct_word_idx(corpus)
        corpus = [[self.word_to_idx[word] for word in sentence] for sentence in corpus]

        #make dataset
        self.x, self.y = self.construct_dataset(corpus, config)

    def pre_process(self, articles):
        print('preprocessing the corpus')
        articles = [pre_process_raw_article(article) for article in articles]
        sentences = itertools.chain.from_iterable([sent_tokenize(article) for article in articles])
        corpus = [mecab_tokenize(s) for s in list(sentences)]

        return corpus

    def construct_word_idx(self, corpus):
        print('constructing word matrix')
        word_set = set(itertools.chain.from_iterable(corpus))
        word_to_idx = {word : idx for idx, word in enumerate(word_set)}
        idx_to_word = {word_to_idx[word] : word for word in word_to_idx}

        return word_to_idx, idx_to_word

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()

        return self.x[idx], self.y[idx]

    def fast_text_pre_process(self, corpus):
        pass

    @abstractmethod
    def construct_dataset(self, corpus, config):
        raise NotImplementedError
