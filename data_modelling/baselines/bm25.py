from rank_bm25 import BM25Okapi
from nltk import word_tokenize
from nltk.corpus import stopwords
import string

def rank_contexts(contexts, target_title, target_lead, mentions = None):
    stop_words = set(stopwords.words('english'))
    punctuation = '!"#&\'()*+,./:;<=>?@[\\]^_`{|}~'
    tokenized_contexts = [word_tokenize(context.lower().replace('\n', ' ').translate(str.maketrans(punctuation, ' '*len(punctuation)))) for context in contexts]
    tokenized_contexts = [[word for word in context if word not in stop_words] for context in tokenized_contexts]
    # tokenized_contexts = [context.lower().replace('\n', ' ').split(" ") for context in contexts]
    bm25 = BM25Okapi(tokenized_contexts)
    # tokenized_target_lead = word_tokenize(target_title.lower().replace('\n', ' ').translate(str.maketrans('', '', string.punctuation)))
    if mentions is None:
        tokenized_query = [target_title.lower()]
        # tokenized_query = [target_title.lower()]
    else:
        # mentions = [mention.lower().split(" ") for mention in mentions]
        # mentions = [word for mention in mentions for word in mention]
        # tokenized_query = target_title.lower().split(" ") + target_lead.lower().replace('\n', ' ').split(" ") + mentions
        mentions = [word_tokenize(mention.lower().replace('\n', ' ').translate(str.maketrans(punctuation, ' '*len(punctuation)))) for mention in mentions]
        mentions = [word for mention in mentions for word in mention]
        tokenized_query = [target_title.lower()] + mentions
        tokenized_query = [word for word in tokenized_query if word not in stop_words]

    scores = bm25.get_scores(tokenized_query)
    return scores
    