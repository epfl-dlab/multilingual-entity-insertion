from openai import OpenAI
import pandas as pd
from nltk import word_tokenize

def get_best_candidate(candidates, source_page, target_page):
    client = OpenAI()
 
    # get the best candidate
    messages=[
        {"role": "system", "content": f"You are a Wikipedia editor. Your task is to find the best span of text in which to insert a link to a target page."},
        {"role": "user", "content": f"The page you are currently editing is titled \"{source_page['source_title']}\". You want to insert a link to the page \"{target_page['target_title']}\" with lead paragraph \"{target_page['target_lead']}\". The text candidates are:"}
    ]
    for candidate in candidates:
        messages.append({"role": "assistant", "content": "Please write the next candidate to consider."})
        messages.append({"role": "user", "content": f"Section Title: {candidate['source_section']}, Text Span: {candidate['link_context']}"})
    messages.append({"role": "assistant", "content": "The most relevant candidate is:"})
    completion = client.chat.completions.create(
        model="gpt-3.5-instruct",
        messages=messages)

    # get the best candidate
    best_candidate = completion.choices[0].message

    # find the index of the most similar candidate
    # measure similarity by cossine similarity
    best_candidate_freqs = {}
    best_candidate_words = 0
    words = word_tokenize(best_candidate)
    for word in words:
        word = word.lower()
        if word in best_candidate_freqs:
            best_candidate_freqs[word] += 1
        else:
            best_candidate_freqs[word] = 1
        best_candidate_words += 1
    
    similarity = 0
    for index, candidate in enumerate(candidates):
        words = word_tokenize(candidate['link_context'])
        candidate_freqs = {}
        candidate_words = 0
        for word in words:
            word = word.lower()
            if word in candidate_freqs:
                candidate_freqs[word] += 1
            else:
                candidate_freqs[word] = 1
            candidate_words += 1
        candidate_similarity = 0
        for word in candidate_freqs:
            if word in best_candidate_freqs:
                candidate_similarity += candidate_freqs[word] * best_candidate_freqs[word]
        candidate_similarity /= candidate_words
        candidate_similarity /= best_candidate_words
        if candidate_similarity > similarity:
            similarity = candidate_similarity
            best_candidate_index = index
            
    return candidates[best_candidate_index], best_candidate_index
            
    
    