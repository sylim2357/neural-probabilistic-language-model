import re
import MeCab

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
