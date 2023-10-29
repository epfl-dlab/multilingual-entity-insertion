def edit_distance(s1, s2):
    """
    Compute the edit distance between two strings.
    """
    m = len(s1)
    n = len(s2)
    dp = [[0 for x in range(n + 1)] for x in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            # If first string is empty, only option is to
            # insert all characters of second string
            if i == 0:
                dp[i][j] = j
            # If second string is empty, only option is to
            # remove all characters of second string
            elif j == 0:
                dp[i][j] = i
            # If last characters are same, ignore last char
            # and recur for remaining string
            elif s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            # If the last character is different, consider all
            # possibilities and find the minimum
            else:
                dp[i][j] = 1 + min(
                    dp[i][j - 1],  # Insert
                    dp[i - 1][j],  # Remove
                    dp[i - 1][j - 1],  # Replace
                )
    return dp[m][n]

def rank_contexts(contexts, target_mentions):
    scores = []
    mentions_words = {}
    contexts_words = []
    for mention in target_mentions:
        words = mention.lower().split()
        if len(words) in mentions_words:
            mentions_words[len(words)].append(words)
        else:
            mentions_words[len(words)] = [words]
    
    for context in contexts:
        lines = context.split('\n')
        words = []
        for line in lines:
            words.extend(line.lower().split())
        temp = {}
        for key in mentions_words:
            temp[key] = []
            # add all key-grams to temp[key]
            for i in range(len(words) - key + 1):
                temp[key].append(words[i:i+key])
        contexts_words.append(temp)
        
    for i in range(len(contexts_words)):
        score = 0
        for key in mentions_words:
            for mention in mentions_words[key]:
                for gram in contexts_words[i][key]:
                    if edit_distance(' '.join(mention), ' '.join(gram)) <= 4:
                        score += 1

        scores.append(score)
    return scores