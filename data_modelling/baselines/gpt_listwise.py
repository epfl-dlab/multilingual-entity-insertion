from openai import OpenAI
import pandas as pd


def get_best_candidate(candidates, target_page, model):
    client = OpenAI()

    messages = [
        {
            'role': 'system',
            'content': 'You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query.'
        },
        # The message below is possibly too long! Do we truncate the lead paragraph?
        {
            'role': 'user',
            'content': f'I will provide you with {{{{{len(candidates)}}}}} passages, each indicated by number identifie []. Rank them based on their relevance to query: {{{{{target_page["target_title"]} {target_page["target_lead"]}}}}}.'
        },
        {
            'role': 'assistant',
            'content': 'Okay, please provide the passages.'
        }
    ]

    for i, candidate in enumerate(candidates):
        messages.append({
            'role': 'user',
            'content': f'[{i+1}] {{{{{candidate["section_title"]} {candidate["link_context"]}}}}}'
        })
        messages.append({
            'role': 'assistant',
            'content': f'Received passage [{i+1}].'
        })

    # This message can also get too long.
    messages.append({
        'role': 'user',
        'content': f'Search Query: {{{{{target_page["target_title"]} {target_page["target_lead"]}}}}}\nRank the {{{{{len(candidates)}}}}} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers, and the most relevant passages should be listed first, and the output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, do not say any word or explain.'
    })

    response = client.send(messages=messages,
                           model=model,
                           temperature=0.2)

    # evaluate the responses
    # need to first see what exactly the response looks like before writing the evaluation code
    raise NotImplementedError