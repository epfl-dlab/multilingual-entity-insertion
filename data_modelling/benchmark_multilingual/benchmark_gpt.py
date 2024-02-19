import argparse
import pandas as pd
import sys
import os
from urllib import parse
from tqdm import tqdm
from ast import literal_eval


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from baselines import gpt_pairwise
from baselines import bm25

def fix_title(title):
    return parse.unquote(title).replace('_', ' ')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the directory containing the data')
    parser.add_argument('--langs', type=str, nargs='+', required=True, help='Languages to benchmark')
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo', help='Which model to use')
    parser.add_argument('--sample_size', type=int, default=100, help='Sample size to use for GPT experiments')
    parser.add_argument('--seed', type=int, default=1234, help='Seed for reproducibility')
    parser.add_argument('--limit_candidates', type=int, default=None, help='Limit the number of candidates to consider')
    parser.add_argument('--column_name', type=str, required=True, help='Name of the column to save the GPT outputs')

    args = parser.parse_args()

    # check if data path exists
    if not os.path.exists(args.data_dir):
        raise ValueError('Data dir does not exist')

    # process each language sequentially
    for lang in args.langs:
        print(f'Processing language {lang}')
        if not os.path.exists(os.path.join(args.data_dir, f'{lang}_gpt.parquet')):
            try:
                df = pd.read_parquet(os.path.join(args.data_dir, f'{lang}.parquet'))
                df = df.sample(n = min(args.sample_size, len(df)), random_state = args.seed).reset_index(drop=True)
                df.to_parquet(os.path.join(args.data_dir, f'{lang}_gpt.parquet'))
            except:
                print(f'No data for language {lang}.')
                continue

        df = pd.read_parquet(os.path.join(args.data_dir, f'{lang}_gpt.parquet'))

        target_titles = df['target_title'].apply(fix_title).tolist()
        target_leads = df['target_lead'].to_list()

        target_pages = [{'target_title': title,
                         'target_lead': lead} for title, lead in zip(target_titles, target_leads)]
        
        df_rows = df.to_dict('records')
        all_contexts = []
        counter = 0
        for i, row in enumerate(tqdm(df_rows)):
            positive_context = {'section_title': row['section'],
                                'link_context': row['context']}
            negative_contexts = literal_eval(row['negative_contexts'])
            row_contexts = []
            if args.limit_candidates:
                # apply BM25 to only include the top (args.limit_candidates - 1) negative contexts
                scores = bm25.rank_contexts([neg['context'] for neg in negative_contexts], target_titles[i], target_leads[i], use_stopwords=True, use_japanese=False)
                scores = [(i, score) for i, score in enumerate(scores)]
                scores = sorted(scores, key=lambda x: x[1], reverse=True)
                negative_contexts = [negative_contexts[i] for i, _ in scores[:args.limit_candidates - 1]]
            for context in negative_contexts:
                negative_context = {'section_title': context['section'],
                                    'link_context': context['context']}
                row_contexts.append([[positive_context, negative_context], [negative_context, positive_context]])
                counter += 2
            all_contexts.append(row_contexts)
    
        print(f'Generated {counter} contexts for language {lang}')
        print('Starting GPT experiments')
        
        gpt_results = []
        n_tokens = 0
        for (target_page, contexts) in (pbar := tqdm(zip(target_pages, all_contexts), total=len(target_pages))):
            gpt_results.append([])
            for i, context_pair in enumerate(contexts):
                pbar.set_description(f'{i} / {len(contexts)} ({n_tokens} tokens)')
                answers = []
                for context in context_pair:
                    result = gpt_pairwise.get_best_candidate(context, target_page, args.model_name)
                    if result == None:
                        print('ERROR')
                        answers.append(None)
                    else:
                        answers.append(result.choices[0].message.content)
                        n_tokens += result.usage.total_tokens
                gpt_results[-1].append(answers)
        print('Finished GPT experiments')
        print('Saving results')
        gpt_results = [str(x) for x in gpt_results]
        df[args.column_name] = gpt_results
        df.to_parquet(os.path.join(args.data_dir, f'{lang}_gpt.parquet'))