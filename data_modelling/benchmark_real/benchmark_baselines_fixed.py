import argparse
import pandas as pd
import sys
import os
from urllib import parse
from tqdm import tqdm
import random
from ast import literal_eval

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from baselines import bm25
from baselines import exact_match
from baselines import fuzzy_match

def fix_title(title):
    return parse.unquote(title).replace('_', ' ')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--mention_map', type=str, required=True)
    parser.add_argument('--data_limit', type=int, default=None,
                        help='Limit the number of rows to use')
    parser.add_argument('--n_contexts', type=int, default=20, help='Number of contexts to use')
    parser.add_argument('--method_name', type=str, required=True, choices=[
                        'random', 'bm25', 'bm25_mentions', 'exact_match', 'fuzzy_match', 'all'], help='Which method to use')
    
    args = parser.parse_args()

    # check if data path exists
    if not os.path.exists(args.data_path):
        raise ValueError('Data path does not exist')
    # check if mention map 
    if not os.path.exists(args.mention_map):
        raise ValueError('Mention map does not exist')

    # load data
    df = pd.read_parquet(args.data_path)
    if args.data_limit is not None and args.data_limit < len(df):
        df = df.sample(args.data_limit)

    source_titles = df['source_title'].apply(fix_title).tolist()
    source_leads = df['source_lead'].tolist()
    target_titles = df['target_title'].apply(fix_title).tolist()
    target_leads = df['target_lead'].tolist()
    contexts = [[] for _ in range(len(df))]
    source_sections = [[] for _ in range(len(df))]
    all_contexts = []
    all_sections = []
    for i, (pos_context, section) in enumerate(zip(df['context'].tolist(), df['section'].tolist())):
        contexts[i].append(pos_context)
        source_sections[i].append(section)
        all_contexts.append(pos_context)
        all_sections.append(section)
    for i, neg_contexts in enumerate(df['negative_contexts'].tolist()):
        neg_contexts = literal_eval(neg_contexts)
        for context in neg_contexts:
            contexts[i].append(context['context'])
            source_sections[i].append(context['section'])
            all_contexts.append(context['context'])
            all_sections.append(context['section'])
    
    mention_map_pre = pd.read_parquet(args.mention_map)
    mention_map_pre = mention_map_pre.to_dict('records')
    mention_map = {}
    for row in mention_map_pre:
        title = fix_title(row['target_title'])
        if title in mention_map:
            mention_map[title].append(row['mention'])
        else:
            mention_map[title] = [row['mention']]
            
    for i in range(len(contexts)):
        if len(contexts[i]) > args.n_contexts:
            # take a sample of contexts and sections
            # they need to be sampled jointly
            sampled_contexts = [contexts[i][0]]
            sampled_sections = [source_sections[i][0]]
            sample_indices = random.sample(range(1, len(contexts[i])), args.n_contexts - 1)
            for index in sample_indices:
                sampled_contexts.append(contexts[i][index])
                sampled_sections.append(source_sections[i][index])
            contexts[i] = sampled_contexts
            source_sections[i] = sampled_sections
        elif len(contexts[i]) < args.n_contexts:
            used_contexts = set(contexts[i])
            while len(contexts[i]) < args.n_contexts:
                random_index = random.randint(0, len(all_contexts) - 1)
                if all_contexts[random_index] in used_contexts:
                    continue
                # check if mentions are not in context
                if target_titles[i] not in mention_map:
                    mention_map[target_titles[i]] = [target_titles[i]]
                if not any([mention.lower() in all_contexts[random_index].lower() for mention in mention_map[target_titles[i]]]):
                    contexts[i].append(all_contexts[random_index])
                    source_sections[i].append(all_sections[random_index])
            
            
    if args.method_name == 'all':
        args.method_name = ['random', 'bm25', 'bm25_mentions', 'exact_match', 'fuzzy_match',]
    else:
        args.method_name = [args.method_name]

    if 'random' in args.method_name:
        print('Calculating random ranks')
        rank = []
        for context in tqdm(contexts):
            position = random.randint(1, len(context))
            rank.append(position)
        df['random_rank'] = rank
    if 'bm25' in args.method_name:
        print('Calculating bm25 ranks')
        rank = []
        for context, title, lead in tqdm(zip(contexts, target_titles, target_leads), total=len(target_titles)):
            scores = bm25.rank_contexts(context, title, lead)
            position = 1
            tied = 0
            if max(scores) < 0:
                scores = [-score for score in scores]
            for i, score in enumerate(scores[1:]):
                if score > scores[0]:
                    position += 1
                elif abs(score - scores[0]) < 0.00001:
                    tied += 1
            rank.append(position + random.randint(0, tied))
        df['bm25_rank'] = rank
    if 'bm25_mentions' in args.method_name:
        print('Calculation bm25 with mention knowledge')
        rank = []
        for context, title, lead in tqdm(zip(contexts, target_titles, target_leads), total=len(target_titles)):
            if title not in mention_map:
                mention_map[title] = [title]
            scores = bm25.rank_contexts(context, title, lead, mention_map[title])
            position = 1
            tied = 0
            if max(scores) < 0:
                scores = [-score for score in scores]
            for i, score in enumerate(scores[1:]):
                if score > scores[0]:
                    position += 1
                elif abs(score - scores[0]) < 0.00001:
                    tied += 1
            rank.append(position + random.randint(0, tied))
        df['bm25_mentions_rank'] = rank
    if 'exact_match' in args.method_name:
        print('Calculating exact match ranks')
        rank = []
        for context, title, lead in tqdm(zip(contexts, target_titles, target_leads), total=len(target_titles)):
            if title not in mention_map:
                mention_map[title] = [title]
            scores = exact_match.rank_contexts(context, mention_map[title])
            position = 1
            equals = 0
            for score in scores[1:]:
                if score > scores[0]:
                    position += 1
                if score == scores[0]:
                    equals += 1
            rank.append(position + random.randint(0, equals))
        df['exact_match_rank'] = rank
    if 'fuzzy_match' in args.method_name:
        print('Calculating fuzzy match ranks')
        rank = []
        for context, title, lead in tqdm(zip(contexts, target_titles, target_leads), total=len(target_titles)):
            if title not in mention_map:
                mention_map[title] = [title]
            scores = fuzzy_match.rank_contexts(context, mention_map[title])
            position = 1
            equals = 0
            for score in scores[1:]:
                if score > scores[0]:
                    position += 1
                if score == scores[0]:
                    equals += 1
            rank.append(position + random.randint(0, equals))
        df['fuzzy_match_rank'] = rank

    df.to_parquet('test_ranking_scores_fixed.parquet')