from dataset.w2v_dataset import W2VDataset
import collections
import itertools


class FastTextDataset(W2VDataset):
    """Fast Text Dataset."""

    def __init__(self, config):
        super().__init__(config)

    def pre_process(articles):
        print('preprocessing the corpus')
        articles = [pre_process_raw_article(article) for article in articles]
        sentences = itertools.chain.from_iterable([sent_tokenize(article) for article in articles])
        corpus = [mecab_tokenize(s) for s in list(sentences)]
        corpus = [[self.ngram(w, 3) for w in s] for s in corpus]

        return corpus

    def fast_text_pre_process(self, corpus):
        return [[self.ngram(w, 3) for w in s] for s in corpus]

    def ngram(self, w, n):
        word = '<' + w + '>'
        if len(word) <= 3:
            return [word]
        else:
            ngram = []
            for i in range(n, len(word)+1):
                ngram += [word[i-n:i]]

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
