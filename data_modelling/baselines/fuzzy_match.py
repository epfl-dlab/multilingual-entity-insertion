from nltk import word_tokenize
from nltk.corpus import stopwords

def rank_contexts(contexts, target_mentions):
    stop_words = set(stopwords.words('english'))
    punctuation = '!"#&\'()*+,./:;<=>?@[\\]^_`{|}~'
    tokenized_contexts = [word_tokenize(context.lower().replace('\n', ' ').translate(str.maketrans(punctuation, ' '*len(punctuation)))) for context in contexts]
    tokenized_contexts = [[word for word in context if word not in stop_words] for context in tokenized_contexts]
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
        scores.append(score / len(context))
    return scores