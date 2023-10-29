from rank_bm25 import BM25Okapi

def rank_contexts(contexts, target_title, target_lead, mentions = None):
    tokenized_contexts = [context.lower().split(" ") for context in contexts]
    bm25 = BM25Okapi(tokenized_contexts)
    if mentions is None:
        tokenized_query = target_title.lower().split(" ") + target_lead.lower().split(" ")
    else:
        mentions = [mention.lower().split(" ") for mention in mentions]
        mentions = [word for mention in mentions for word in mention]
        tokenized_query = target_title.lower().split(" ") + target_lead.lower().split(" ") + mentions

    scores = bm25.get_scores(tokenized_query)
    return scores
    