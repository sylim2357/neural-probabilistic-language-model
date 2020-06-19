import easydict
import torch
import MeCab
import json
import re


def pre_process_raw_article(article):
    """Pre-processing news articles.

    Args
        article (str): article text

    Return
        artile (str): processed text

    """
    replacements = [
        ('[“”]', '"'),
        ('[‘’]', '\''),
        ('\([^)]*\)', ''),
        ('[^가-힣\'"A-Za-z0-9.\s\?\!]', ' '),
        ('(?=[^0-9])\.(?=[^0-9])', '. '),
        ('\s\s+', ' ')
    ]

    for old, new in replacements:
        article = re.sub(old, new, article)

    return article

def mecab_tokenize(sentence):
    t = MeCab.Tagger()
    return [re.split(',', re.sub('\t', ',', s))[0] for s in t.parse(sentence).split('\n') if (s!='') & ('EOS' not in s)]

def collate_fn(data):
    seqs, labels = zip(*data)
    return seqs, labels

def config_parser(args):
    print('file path is ' + str(args.file_path))
    with open(args.config_path, 'rb') as f:
        config = easydict.EasyDict(json.load(f))
    config.model = args.model
    config.file_path = args.file_path
    config.dataset_path = args.dataset_path
    config.device = torch.device(args.device)
    return config

def ngram(self, w, n):
    word = '<' + w + '>'
    if len(word) <= 3:
        return [word]
    else:
        ngram = []
        for i in range(n, len(word)+1):
            ngram += [word[i-n:i]]

        return ngram + [word]
