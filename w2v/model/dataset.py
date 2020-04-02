# from w2v.utils import pre_process_raw_article, mecab_tokenize
from utils import pre_process_raw_article, mecab_tokenize
from torch.utils.data import Dataset
from nltk import sent_tokenize
from abc import abstractmethod
import pandas as pd
import itertools


class W2VDataset(Dataset):
    def __init__(self, config):
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

        @abstractmethod
        def construct_dataset(self, corpus, config):
            raise NotImplementedError

class CBOWDataset(W2VDataset):
    """CBOW Dataset.

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
        super().__init__(config)

    def construct_dataset(self, corpus, config):
        print('constructing training dataset')
        x, y = [], []
        for sentence in corpus:
            for i in range(config.window_size, len(sentence) - config.window_size):
                x += [sentence[i-config.window_size:i] + sentence[i+1:i+condig.window_size+1]]
                y += [sentence[i]]
        return x, y

class SkipGramDataset(W2VDataset):
    """Skip-Gram Dataset.

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
        super().__init__(config)

    def construct_dataset(self, corpus, config):
        print('constructing training dataset')
        x, y = [], []
        for sentence in corpus:
            for i in range(config.window_size, len(sentence) - config.window_size):
                x += [sentence[i]] * (config.window_size*2)
                y += sentence[i-config.window_size:i] + sentence[i+1:i+condig.window_size+1]
        return x, y

class NegSamplingDataset(W2VDataset):
    """Negative Sampling Dataset.

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
        super().__init__(config)

    def construct_dataset(self, corpus, config):
        print('constructing training dataset')
        x, y = [], []
        for sentence in corpus:
            for i in range(config.window_size, len(sentence) - config.window_size):
                x += [sentence[i]] * (config.window_size*2)
                y += sentence[i-config.window_size:i] + sentence[i+1:i+condig.window_size+1]
                z += random.sample(idx_to_word.keys() - set(sentence[i-config.window_size:i+condig.window_size+1]), config.window_size*2)
        return x, y, z
