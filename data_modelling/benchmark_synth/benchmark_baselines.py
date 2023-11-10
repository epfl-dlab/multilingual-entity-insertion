import argparse
import pandas as pd
import sys
import os
from urllib import parse
from tqdm import tqdm
import random

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from baselines import bm25
from baselines import exact_match
from baselines import fuzzy_match

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--mention_map', type=str, required=True)
    parser.add_argument('--data_limit', type=int, default=None,
                        help='Limit the number of rows to use')
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

    source_titles = df['source_title'].tolist()
    source_leads = df['source_lead'].tolist()
    target_titles = df['target_title'].tolist()
    target_leads = df['target_lead'].tolist()
    contexts = [[] for _ in range(len(target_titles))]
    source_sections = [[] for _ in range(len(target_titles))]
    all_sections = set([])
    for column in df:
        if 'link_context' in column:
            for i, context in enumerate(df[column].tolist()):
                contexts[i].append(context)
        if 'source_section' in column:
            for i, source_section in enumerate(df[column].tolist()):
                source_sections[i].append(source_section)
                all_sections.add(source_section)
    all_sections = list(all_sections)

    mention_map_pre = pd.read_parquet(args.mention_map)
    mention_map_pre = mention_map_pre.to_dict('records')
    mention_map = {}
    for row in mention_map_pre:
        title = parse.unquote(row['target_title']).replace('_', ' ')
        if title in mention_map:
            mention_map[title].append(row['mention'])
        else:
            mention_map[title] = [row['mention']]
            
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
            for score in scores[1:]:
                if score > scores[0]:
                    position += 1
            rank.append(position)
        df['bm25_rank'] = rank
    if 'bm25_mentions' in args.method_name:
        print('Calculation bm25 with mention knowledge')
        rank = []
        for context, title, lead in tqdm(zip(contexts, target_titles, target_leads), total=len(target_titles)):
            if title not in mention_map:
                mention_map[title] = [title]
            scores = bm25.rank_contexts(context, title, lead, mention_map[title])
            position = 1
            for score in scores[1:]:
                if score > scores[0]:
                    position += 1
            rank.append(position)
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

    df.to_parquet('test_ranking_scores.parquet')