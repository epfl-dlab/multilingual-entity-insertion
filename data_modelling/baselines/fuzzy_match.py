from nltk import word_tokenize
from nltk.corpus import stopwords
import MeCab

def ja_tokenize(text):
    wakati = MeCab.Tagger("-Owakati")
    tokens = wakati.parse(text).split()
    return tokens    


def rank_contexts(contexts, target_mentions, use_stopwords = True, use_japanese = False):
    if use_stopwords:
        stop_words = set(stopwords.words('english'))
    else:
        stop_words = []
    punctuation = '!"#&\'()*+,./:;<=>?@[\\]^_`{|}~'
    if use_japanese:
        tokenized_contexts = [ja_tokenize(context.lower().replace('\n', ' ').translate(str.maketrans(punctuation, ' '*len(punctuation)))) for context in contexts]
    else:
        tokenized_contexts = [word_tokenize(context.lower().replace('\n', ' ').translate(str.maketrans(punctuation, ' '*len(punctuation)))) for context in contexts]
    tokenized_contexts = [[word for word in context if word not in stop_words] for context in tokenized_contexts]
    if use_japanese:
        tokenized_mentions = [ja_tokenize(mention.lower().replace('\n', ' ').translate(str.maketrans(punctuation, ' '*len(punctuation)))) for mention in target_mentions]
    else:
        tokenized_mentions = [word_tokenize(mention.lower().replace('\n', ' ').translate(str.maketrans(punctuation, ' '*len(punctuation)))) for mention in target_mentions]
    tokenized_mentions = [[word for word in mention if word not in stop_words] for mention in tokenized_mentions]
    tokenized_mentions = [word for mention in tokenized_mentions for word in mention]

    scores = []
    freqs = {}
    for word in tokenized_mentions:
        freqs[word] = freqs.get(word, 0) + 1
    for context in tokenized_contexts:
        score = 0
        for word in context:
            score += freqs.get(word, 0)
        if len(context) == 0:
            scores.append(score)
        else:
            scores.append(score / len(context))
    return scores