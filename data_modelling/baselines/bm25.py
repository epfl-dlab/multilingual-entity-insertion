from rank_bm25 import BM25Okapi
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
import MeCab

def ja_tokenize(text):
    wakati = MeCab.Tagger("-Owakati")
    tokens = wakati.parse(text).split()
    return tokens    

def rank_contexts(contexts, target_title, target_lead, mentions = None, use_stopwords = True, use_japanese = False):
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
    bm25 = BM25Okapi(tokenized_contexts)
    if use_japanese:
        tokenized_target_lead = ja_tokenize(target_lead.lower().replace('\n', ' ').translate(str.maketrans('', '', string.punctuation)))
    else:
        tokenized_target_lead = word_tokenize(target_lead.lower().replace('\n', ' ').translate(str.maketrans('', '', string.punctuation)))
    tokenized_target_lead = [word for word in tokenized_target_lead if word not in stop_words]
    if mentions is None:
        tokenized_query = [target_title.lower()] + tokenized_target_lead
    else:
        if use_japanese:
            mentions = [ja_tokenize(mention.lower().replace('\n', ' ').translate(str.maketrans(punctuation, ' '*len(punctuation)))) for mention in mentions]
        else:
            mentions = [word_tokenize(mention.lower().replace('\n', ' ').translate(str.maketrans(punctuation, ' '*len(punctuation)))) for mention in mentions]
        mentions = [word for mention in mentions for word in mention if word not in stop_words]        
        tokenized_query = [target_title.lower()] + mentions + tokenized_target_lead

    scores = bm25.get_scores(tokenized_query)
    return scores
    