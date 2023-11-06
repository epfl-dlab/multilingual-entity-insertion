def rank_contexts(contexts, target_mentions):
    scores = []
    freqs = {}
    for mention in target_mentions:
        words = [word.lower() for word in mention.split()]
        for word in words:
            freqs[word] = freqs.get(word, 0) + 1
    for context in contexts:
        score = 0
        for word in context.split():
            score += freqs.get(word.lower(), 0)
        scores.append(score / len(context))
    return scores