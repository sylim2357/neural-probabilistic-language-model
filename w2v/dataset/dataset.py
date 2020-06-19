from utils import pre_process_raw_article, mecab_tokenize
from torch.utils.data import Dataset
from nltk import sent_tokenize
from abc import abstractmethod
import pandas as pd
import collections
import itertools
import random


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

    @abstractmethod
    def construct_dataset(self, corpus, config):
        raise NotImplementedError

class CBOWDataset(W2VDataset):
    """CBOW Dataset"""

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
    """Skip-Gram Dataset"""

    def __init__(self, config):
        super().__init__(config)

    def construct_dataset(self, corpus, config):
        print('constructing training dataset')
        x, y = [], []
        for sentence in corpus:
            for i in range(config.window_size, len(sentence) - config.window_size):
                x += [sentence[i]] * (config.window_size*2)
                y += sentence[i-config.window_size:i] + sentence[i+1:i+config.window_size+1]

        return x, y

class NegSamplingDataset(W2VDataset):
    """Negative Sampling Dataset.

    Args:
        config (dict): hyperparameters
        word_frequency (dict): word index - word frequency map for negative sampling

    """

    def __init__(self, config):
        super().__init__(config)

    def construct_word_idx(self, corpus):
        print('constructing word matrix')
        word_frequency = collections.Counter(itertools.chain.from_iterable(corpus))
        word_frequency = {word: word_frequency[word]**(3/4) for idx, word in enumerate(word_frequency)}
        word_to_idx = {word: idx for idx, word in enumerate(word_frequency)}
        idx_to_word = {word_to_idx[word]: word for word in word_to_idx}
        self.word_frequency = {word_to_idx[word]: word_frequency[word] for word in word_frequency}

        return word_to_idx, idx_to_word

    def construct_dataset(self, corpus, config):
        print('constructing training dataset')
        target, pos, neg = [], [], []
        for sentence in corpus:
            for i in range(config.window_size, len(sentence) - config.window_size):
                target += [sentence[i]] * (config.window_size*2)
                pos += sentence[i-config.window_size:i] + sentence[i+1:i+config.window_size+1]
                neg.append(self.neg_sample(sentence[i-config.window_size:i+config.window_size+1], config))
        neg = np.array(neg).reshape(-1, config.window_size*2)

        return (pos, neg), target

    def neg_sample(self, word_contxt, config):
        word_universe = self.idx_to_word.keys() - set(word_contxt)
        word_distn = np.array([self.word_frequency[idx] for idx in word_universe])
        word_distn = word_distn / word_distn.sum()

        return np.random.choice(a=list(word_universe), size=config.neg_sample_size*config.window_size*2, p=word_distn)

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()

        return [self.x[0][idx], self.y[idx], 1], [self.x[1][idx], self.y[idx], 0]

class FastTextDataset(W2VDataset):
    """Fast Text Dataset.

    Args:
        config (dict): hyperparameters
        word_frequency (dict): word index - word frequency map for negative sampling

    """

    def __init__(self, config):
        super().__init__(config)

    def pre_process(self, articles):
        print('preprocessing the corpus')
        articles = [pre_process_raw_article(article) for article in articles]
        sentences = itertools.chain.from_iterable([sent_tokenize(article) for article in articles])
        corpus = [mecab_tokenize(s) for s in list(sentences)]
        corpus = [self.ngram(w, 3) for w in corpus]

        return corpus

    def ngram(w, n):
        ngram = []
        word = '<' + w + '>'
        for i in range(n, len(word)-1):
            ngram += word[i-n:i]

        return ngram + [word]

    def construct_word_idx(self, corpus):
        print('constructing word matrix')
        word_frequency = collections.Counter(itertools.chain.from_iterable(itertools.chain.from_iterable(corpus)))
        word_frequency = {word: word_frequency[word]**(3/4) for idx, word in enumerate(word_frequency)}
        word_to_idx = {word: idx for idx, word in enumerate(word_frequency)}
        idx_to_word = {word_to_idx[word]: word for word in word_to_idx}
        self.word_frequency = {word_to_idx[word]: word_frequency[word] for word in word_frequency}

        return word_to_idx, idx_to_word

    def construct_dataset(self, corpus, config):
        print('constructing training dataset')
        target, pos, neg = [], [], []
        for sentence in corpus:
            for i in range(config.window_size, len(sentence) - config.window_size):
                target += [sentence[i]] * (config.window_size*2)
                pos += sentence[i-config.window_size:i] + sentence[i+1:i+config.window_size+1]
                neg.append(self.neg_sample(sentence[i-config.window_size:i+config.window_size+1], config))
        neg = np.array(neg).reshape(-1, config.window_size*2)

        return (pos, neg), target

    def neg_sample(self, word_contxt, config):
        word_universe = self.idx_to_word.keys() - set(word_contxt)
        word_distn = np.array([self.word_frequency[idx] for idx in word_universe])
        word_distn = word_distn / word_distn.sum()

        return random.choice(list(word_universe), config.neg_sample_size*config.window_size*2)

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()

        return [self.x[0][idx], self.y[idx], 1], [self.x[1][idx], self.y[idx], 0]
