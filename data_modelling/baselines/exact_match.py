def rank_contexts(contexts, target_mentions):
    scores = []
    for context in contexts:
        score = 0
        for mention in target_mentions:
            score += context.lower().count(mention.lower())
        scores.append(score / len(context))
    return scores