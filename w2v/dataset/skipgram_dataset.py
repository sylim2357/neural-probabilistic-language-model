from dataset.w2v_dataset import W2VDataset


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
