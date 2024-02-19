from openai import OpenAI
import openai
import pandas as pd
from dotenv import load_dotenv
from time import sleep


def get_best_candidate(candidates, target_page, model, try_num=0, max_tries=3, prompt_type=1):
    if try_num >= max_tries:
        return None
    
    load_dotenv()
    client = OpenAI()
    
    if prompt_type == 1:
        # base prompt
        messages = [
            {
                'role': 'system',
                'content': 'You are EntityGPT, an intelligent assistant that can select passages in which to insert mentions to a query entity.'
            },
            {
                'role': 'user',
                'content': f'I will give you a JSON document with {{{{2}}}} passages, Passage A and Passage B, and a query entity, defined by the entity\'s name and a short description. ' + \
                            'Your task is to the find the passage most relevant for inserting a mention to the query entity.'
            },
            {
                'role': 'assistant',
                'content': 'Okay, please provide me with the passages and the query entity.'
            },
            {
                'role': 'user',
                'content': f'{{"Passage A": {candidates[0]["section_title"]} {candidates[0]["link_context"]},\n' + \
                        f' "Passage B": {candidates[1]["section_title"]} {candidates[1]["link_context"]},\n' + \
                        f' "Query Entity": {target_page["target_title"]} {target_page["target_lead"]}}}'
            },
            {
                'role': 'assistant',
                'content': 'Received the JSON document.'
            },
            {
                'role': 'user',
                'content': f'Select the passage most relevant for inserting a mention to the query entity. ' + \
                        f'Only answer either {{{{Passage A}}}} or {{{{Passage B}}}}, do not explain your choice or provide any additional answer.'
            }
        ]
    
    if prompt_type == 2:
        # prompt with more separated knowledge
        messages = [
            {
                'role': 'system',
                'content': 'You are EntityGPT, an intelligent assistant that can select passages in which to insert mentions to a query entity.'
            },
            {
                'role': 'user',
                'content': f'I will give you a query entity and two passage, Passage A and Passage B. ' + \
                            'Your task is to find the passage most relevant for inserting a mention to the query entity.'
            },
            {
                'role': 'assistant',
                'content': 'Okay, please provide me with the query entity.'
            },
            {
                'role': 'user',
                'content': f'{{{{Query Entity}}}}: {target_page["target_title"]} {target_page["target_lead"]}'
            },
            {
                'role': 'assistant',
                'content': 'Received the query entity. Please provide me with passage A.'
            },
            {
                'role': 'user',
                'content': f'{{{{Passage A}}}}: {candidates[0]["section_title"]} {candidates[0]["link_context"]}'
            },
            {
                'role': 'assistant',
                'content': 'Received passage A. Please provide me with passage B.'
            },
            {
                'role': 'user',
                'content': f'{{{{Passage B}}}}: {candidates[1]["section_title"]} {candidates[1]["link_context"]}'
            },
            {
                'role': 'assistant',
                'content': 'Received passage B.'
            },
            {
                'role': 'user',
                'content': f'Select the passage most relevant for inserting a mention to the query entity. ' + \
                        f'Only answer either {{{{Passage A}}}} or {{{{Passage B}}}}, do not explain your choice or provide any additional answer.'
            }
        ]
        
    if prompt_type == 3:
        # prompt almost identical to RankGPT
        messages = [
            {
                'role': 'system',
                'content': 'You are RelevanceGPT, an intelligent assistant that can select passages based on their relevancy to the query.'
            },
            {
                'role': 'user',
                'content': f'I will provide you with 2 passages, Passage A and Passage B. ' + \
                           f'Select the best passage based on their relevance to query: {{{{{target_page["target_title"]} {target_page["target_lead"]}}}}}.'
            },
            {
                'role': 'assistant',
                'content': 'Okay, please provide me with the passages.'
            },
            {
                'role': 'user',
                'content': f'Passage A: {{{{{candidates[0]["section_title"]} {candidates[0]["link_context"]}}}}}'
            },
            {
                'role': 'assistant',
                'content': 'Received passage A.'
            },
            {
                'role': 'user',
                'content': f'Passage B: {{{{{candidates[1]["section_title"]} {candidates[1]["link_context"]}}}}}'
            },
            {
                'role': 'assistant',
                'content': 'Received passage B.'
            },
            {
                'role': 'user',
                'content': f'Search Query: {{{{{target_page["target_title"]} {target_page["target_lead"]}}}}}.\n' + \
                           f'Select the best passage based on their relevance to the search query. Only response either {{{{Passage A}}}} or {{{{Passage B}}}}, do not explain your choice or provide any additional answer.'
            }
        ]
        
    if prompt_type == '4':
        # prompt with only the target title
        messages = [
            {
                'role': 'system',
                'content': 'You are EntityGPT, an intelligent assistant that can select passages in which to insert mentions to a query entity.'
            },
            {
                'role': 'user',
                'content': f'I will give you a query entity and two passage, Passage A and Passage B. ' + \
                            'Your task is to find the passage most relevant for inserting a mention to the query entity.'
            },
            {
                'role': 'assistant',
                'content': 'Okay, please provide me with the query entity.'
            },
            {
                'role': 'user',
                'content': f'{{{{Query Entity}}}}: {target_page["target_title"]}'
            },
            {
                'role': 'assistant',
                'content': 'Received the query entity. Please provide me with passage A.'
            },
            {
                'role': 'user',
                'content': f'{{{{Passage A}}}}: {candidates[0]["section_title"]} {candidates[0]["link_context"]}'
            },
            {
                'role': 'assistant',
                'content': 'Received passage A. Please provide me with passage B.'
            },
            {
                'role': 'user',
                'content': f'{{{{Passage B}}}}: {candidates[1]["section_title"]} {candidates[1]["link_context"]}'
            },
            {
                'role': 'assistant',
                'content': 'Received passage B.'
            },
            {
                'role': 'user',
                'content': f'Select the passage most relevant for inserting a mention to the query entity. ' + \
                        f'Only answer either {{{{Passage A}}}} or {{{{Passage B}}}}, do not explain your choice or provide any additional answer.'
            }
        ]
        
    if prompt_type == '5':
        # prompt with only the target title in its canonical form
        messages = [
            {
                'role': 'system',
                'content': 'You are EntityGPT, an intelligent assistant that can select passages in which to insert mentions to a query entity.'
            },
            {
                'role': 'user',
                'content': f'I will give you a query entity and two passage, Passage A and Passage B. ' + \
                            'Your task is to find the passage most relevant for inserting a mention to the query entity.'
            },
            {
                'role': 'assistant',
                'content': 'Okay, please provide me with the query entity.'
            },
            {
                'role': 'user',
                'content': f'{{{{Query Entity}}}}: {target_page["target_title"].replace(" ", "_")}'
            },
            {
                'role': 'assistant',
                'content': 'Received the query entity. Please provide me with passage A.'
            },
            {
                'role': 'user',
                'content': f'{{{{Passage A}}}}: {candidates[0]["section_title"]} {candidates[0]["link_context"]}'
            },
            {
                'role': 'assistant',
                'content': 'Received passage A. Please provide me with passage B.'
            },
            {
                'role': 'user',
                'content': f'{{{{Passage B}}}}: {candidates[1]["section_title"]} {candidates[1]["link_context"]}'
            },
            {
                'role': 'assistant',
                'content': 'Received passage B.'
            },
            {
                'role': 'user',
                'content': f'Select the passage most relevant for inserting a mention to the query entity. ' + \
                        f'Only answer either {{{{Passage A}}}} or {{{{Passage B}}}}, do not explain your choice or provide any additional answer.'
            }
        ]
        
        

    with open('error.log', 'a') as ferr:
        
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=1,
                seed=1234,
                max_tokens=100,
                top_p=1,
                timeout=5
            )
        except (openai.RateLimitError, openai.APIError, openai.APITimeoutError, openai.InternalServerError) as e1:
                ferr.write(f'Encountered an exception {e1}, sleeping\n')
                ferr.write(f'Retrying. Current attempt = {try_num}\n')
                sleep(10)
                completion = get_best_candidate(candidates, target_page, model, try_num=try_num+1, max_tries=max_tries)
        except ValueError as e2:
            ferr.write(f'Encountered an exception {e2}\n')
            ferr.write(f'Retrying. Current attempt = {try_num}\n')
            completion = get_best_candidate(candidates, target_page, model, try_num=try_num+1, max_tries=max_tries)

        return completion