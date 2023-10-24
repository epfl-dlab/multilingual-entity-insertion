from rank_bm25 import BM25Okapi

def rank_contexts(contexts, source_title, source_lead):
    tokenized_contexts = [context.split(" ") for context in contexts]
    bm25 = BM25Okapi(tokenized_contexts)
    tokenized_query = source_title.split(" ") + source_lead.split(" ")
    
    scores = bm25.get_scores(tokenized_query)
    return scores
    