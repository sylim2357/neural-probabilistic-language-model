from torch.utils.data import Dataset
import pandas as pd

class NLPCorpusDataset(Dataset):
    """NLP Corpus dataset.

    Args:
        csv_file (str): path to the csv file
        root_dir (str): root
        args (dict): hyperparameters

    Attributes:
        root_dir (str): root
        word_to_idx (dict): word_to_idx mapping
        idx_to_word (dict): idx_to_word mapping
        x (list): train data (5-gram)
        y (list): label

    """

    def __init__(self, csv_file, root_dir, args):
        articles = pd.read_csv(csv_file, encoding='utf-8')['article'].dropna().values
        articles = [pre_process_raw_article(article) for article in articles]
        sentences = itertools.chain.from_iterable([sent_tokenize(article) for article in articles])
        corpus = [mecab_tokenize(s) for s in list(sentences)]
        self.root_dir = root_dir
        del articles
        del sentences

        #construct word matrix
        word_set = set(itertools.chain.from_iterable(corpus))
        self.word_to_idx = {word : idx for idx, word in enumerate(word_set)}
        self.idx_to_word = {self.word_to_idx[word] : word for word in self.word_to_idx}
        del word_set
        corpus = [[self.word_to_idx[word] for word in sentence] for sentence in corpus]

        #make train label dataset
        self.x = []
        self.y = []
        for sentence in corpus:
            for i in range(len(sentence) - args.window_size):
                self.x.append(sentence[i:i+args.window_size])
                self.y.append([sentence[i+args.window_size]])
        del corpus

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.x[idx], self.y[idx]
