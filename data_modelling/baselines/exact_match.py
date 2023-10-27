def rank_contexts(contexts, target_mentions):
    scores = []
    for context in contexts:
        score = 0
        for mention in target_mentions:
            if mention in context:
                score += 1
        scores.append(score / len(context))
    return scores