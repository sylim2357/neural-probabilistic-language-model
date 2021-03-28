from dataset.negsampling_dataset import NegSamplingDataset
from utils import pre_process_raw_article, mecab_tokenize
from torch.utils.data import Dataset
from nltk import sent_tokenize
from abc import abstractmethod
import pandas as pd
import collections
import itertools


class FastTextDataset(NegSamplingDataset):
    """Fast Text Dataset.

    Args:
        config (dict): hyperparameters
        word_frequency (dict): word index - word frequency map for negative sampling

    """

    def __init__(self, config):
        if 'pkl' in config.file_path:
            with open(config.file_path, 'rb') as f:
                corpus = pickle.load(f)[:1000]
        elif config.file_path == 'treebank':
            corpus = treebank.sents()
        else:
            articles = (
                pd.read_csv(config.file_path, encoding='utf-8')['article']
                .dropna()
                .values
            )
            # pre process
            corpus = self.pre_process(articles)

        ngram_corpus = self.fast_text_pre_process(corpus)
        (
            self.ngram_word_to_idx,
            self.ngram_idx_to_word,
            _,
        ) = self.construct_word_idx(ngram_corpus)
        # construct word matrix
        (
            self.word_to_idx,
            self.idx_to_word,
            self.word_frequency,
        ) = self.construct_word_idx(corpus)
        # make dataset
        self.x, self.y = self.construct_dataset(corpus, config)

    def ngram(self, w, n):
        word = '<' + w + '>'
        if len(word) <= 3:
            return [word]
        else:
            ngram = []
            for i in range(n, len(word) + 1):
                ngram += [word[i - n : i]]

            return ngram + [word]

    def fast_text_pre_process(self, corpus):
        return [[self.ngram(w, 3) for w in s] for s in corpus]

    def construct_word_idx(self, corpus):
        print('constructing word matrix')
        corpus_flatten = list(itertools.chain.from_iterable(corpus))
        if isinstance(corpus_flatten[0], list):
            word_frequency = collections.Counter(
                itertools.chain.from_iterable(corpus_flatten)
            )
        else:
            word_frequency = collections.Counter(corpus_flatten)
        word_frequency = {
            word: word_frequency[word] ** (3 / 4)
            for idx, word in enumerate(word_frequency)
        }
        word_to_idx = {word: idx for idx, word in enumerate(word_frequency)}
        idx_to_word = {word_to_idx[word]: word for word in word_to_idx}

        return word_to_idx, idx_to_word, word_frequency

    def neg_sample(self, word_contxt, config):
        word_universe = self.word_to_idx.keys() - set(word_contxt)
        word_distn = np.array(
            [self.word_frequency[idx] for idx in word_universe]
        )
        word_distn = word_distn / word_distn.sum()

        return np.random.choice(
            a=list(word_universe),
            size=config.neg_sample_size * config.window_size * 2,
            p=word_distn,
        )
