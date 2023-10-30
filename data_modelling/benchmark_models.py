import argparse
import pandas as pd
import numpy
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
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
    parser.add_argument('--mention_map_month_1', type=str, required=True)
    parser.add_argument('--mention_map_month_2', type=str, required=True)
    parser.add_argument('--data_limit', type=int, default=None,
                        help='Limit the number of rows to use')
    parser.add_argument('--method_name', type=str, required=True, choices=[
                        'random', 'bm25', 'bm25_mentions', 'exact_match', 'fuzzy_match', 'model_rank_corruption', 'model_rank_no_corruption', 'model_random_section', 'all'], help='Which method to use')
    
    args = parser.parse_args()

    # check if data path exists
    if not os.path.exists(args.data_path):
        raise ValueError('Data path does not exist')
    # check if mention map month 1 exists
    if not os.path.exists(args.mention_map_month_1):
        raise ValueError('Mention map month 1 does not exist')
    # check if mention map month 2 exists
    if not os.path.exists(args.mention_map_month_2):
        raise ValueError('Mention map month 2 does not exist')

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

    mention_map_pre = pd.concat([pd.read_parquet(args.mention_map_month_1), pd.read_parquet(
        args.mention_map_month_2)]).drop_duplicates().reset_index(drop=True)
    mention_map_pre = mention_map_pre.to_dict('records')
    mention_map = {}
    for row in mention_map_pre:
        title = parse.unquote(row['target_title']).replace('_', ' ')
        if title in mention_map:
            mention_map[title].append(row['mention'])
        else:
            mention_map[title] = [row['mention']]
            
    if args.method_name == 'all':
        args.method_name = ['random', 'bm25', 'bm25_mentions', 'exact_match', 'fuzzy_match', 'model_rank_corruption', 'model_rank_no_corruption', 'model_random_section']
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
    if 'model_rank_corruption' in args.method_name:
        print('Calculating model with ranking losses and corruption ranks')
        model_path = 'training_ranking/models/listsoftmax_corruption/model/'
        tokenizer_path = 'training_ranking/models/listsoftmax_corruption/tokenizer/'
        classification_head_path = 'training_ranking/models/listsoftmax_corruption/classification_head.pth'
        column_name = 'model_ranking_corruption_rank'
        model = AutoModel.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        classification_head = nn.Sequential(nn.Linear(model.config.hidden_size * 3, model.config.hidden_size),
                                            nn.ReLU(),
                                            nn.Linear(model.config.hidden_size, 1))
        classification_head.load_state_dict(torch.load(classification_head_path, map_location=torch.device('cpu')))
        model.eval()

        model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
        classification_head = classification_head.to('cuda' if torch.cuda.is_available() else 'cpu')

        rank = []
        with torch.no_grad():
            for context, source_section, source_title, source_lead, target_title, target_lead in tqdm(zip(contexts, source_sections, source_titles, source_leads, target_titles, target_leads), total=len(target_titles)):
                source = tokenizer([f"{source_title}{tokenizer.sep_token}{source_lead}"], return_tensors='pt', padding=True, truncation=True, max_length=256).to('cuda' if torch.cuda.is_available() else 'cpu')
                target = tokenizer([f"{target_title}{tokenizer.sep_token}{target_lead}"], return_tensors='pt', padding=True, truncation=True, max_length=256).to('cuda' if torch.cuda.is_available() else 'cpu')
                source_embeddings = model(**source)['last_hidden_state'][:, 0, :]
                target_embeddings = model(**target)['last_hidden_state'][:, 0, :]
                scores = []
                for c, s in zip(context, source_section):
                    input = tokenizer([f"{s}{tokenizer.sep_token}{c}"], return_tensors='pt', padding=True, truncation=True, max_length=256).to('cuda' if torch.cuda.is_available() else 'cpu')
                    input_embeddings = model(**input)['last_hidden_state'][:, 0, :]
                    input = torch.cat((source_embeddings, input_embeddings, target_embeddings), dim=1)
                    score = classification_head(input)[0].squeeze().item()
                    scores.append(score)
                position = 1
                for score in scores[1:]:
                    if score > scores[0]:
                        position += 1
                rank.append(position)
        df[column_name] = rank
    if 'model_rank_no_corruption' in args.method_name:
        print('Calculating model with ranking losses and no corruption ranks')
        model_path = 'training_ranking/models/listsoftmax_no_corruption/model/'
        tokenizer_path = 'training_ranking/models/listsoftmax_no_corruption/tokenizer/'
        classification_head_path = 'training_ranking/models/listsoftmax_no_corruption/classification_head.pth'
        column_name = 'model_ranking_no_corruption_rank'
        model = AutoModel.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        classification_head = nn.Sequential(nn.Linear(model.config.hidden_size * 3, model.config.hidden_size),
                                            nn.ReLU(),
                                            nn.Linear(model.config.hidden_size, 1))
        classification_head.load_state_dict(torch.load(classification_head_path, map_location=torch.device('cpu')))
        model.eval()

        model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
        classification_head = classification_head.to('cuda' if torch.cuda.is_available() else 'cpu')

        rank = []
        with torch.no_grad():
            for context, source_section, source_title, source_lead, target_title, target_lead in tqdm(zip(contexts, source_sections, source_titles, source_leads, target_titles, target_leads), total=len(target_titles)):
                source = tokenizer([f"{source_title}{tokenizer.sep_token}{source_lead}"], return_tensors='pt', padding=True, truncation=True, max_length=256).to('cuda' if torch.cuda.is_available() else 'cpu')
                target = tokenizer([f"{target_title}{tokenizer.sep_token}{target_lead}"], return_tensors='pt', padding=True, truncation=True, max_length=256).to('cuda' if torch.cuda.is_available() else 'cpu')
                source_embeddings = model(**source)['last_hidden_state'][:, 0, :]
                target_embeddings = model(**target)['last_hidden_state'][:, 0, :]
                scores = []
                for c, s in zip(context, source_section):
                    input = tokenizer([f"{s}{tokenizer.sep_token}{c}"], return_tensors='pt', padding=True, truncation=True, max_length=256).to('cuda' if torch.cuda.is_available() else 'cpu')
                    input_embeddings = model(**input)['last_hidden_state'][:, 0, :]
                    input = torch.cat((source_embeddings, input_embeddings, target_embeddings), dim=1)
                    score = classification_head(input)[0].squeeze().item()
                    scores.append(score)
                position = 1
                for score in scores[1:]:
                    if score > scores[0]:
                        position += 1
                rank.append(position)
        df[column_name] = rank
    if 'model_random_section' in args.method_name:
        print('Calculating model with ranking losses and corruption ranks, but using a random section name')
        model_path = 'training_ranking/models/listsoftmax_corruption/model/'
        tokenizer_path = 'training_ranking/models/listsoftmax_corruption/tokenizer/'
        classification_head_path = 'training_ranking/models/listsoftmax_corruption/classification_head.pth'
        column_name = 'model_random_section_rank'
        model = AutoModel.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        classification_head = nn.Sequential(nn.Linear(model.config.hidden_size * 3, model.config.hidden_size),
                                            nn.ReLU(),
                                            nn.Linear(model.config.hidden_size, 1))
        classification_head.load_state_dict(torch.load(classification_head_path, map_location=torch.device('cpu')))
        model.eval()

        model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
        classification_head = classification_head.to('cuda' if torch.cuda.is_available() else 'cpu')

        rank = []
        with torch.no_grad():
            for context, source_section, source_title, source_lead, target_title, target_lead in tqdm(zip(contexts, source_sections, source_titles, source_leads, target_titles, target_leads), total=len(target_titles)):
                source = tokenizer([f"{source_title}{tokenizer.sep_token}{source_lead}"], return_tensors='pt', padding=True, truncation=True, max_length=256).to('cuda' if torch.cuda.is_available() else 'cpu')
                target = tokenizer([f"{target_title}{tokenizer.sep_token}{target_lead}"], return_tensors='pt', padding=True, truncation=True, max_length=256).to('cuda' if torch.cuda.is_available() else 'cpu')
                source_embeddings = model(**source)['last_hidden_state'][:, 0, :]
                target_embeddings = model(**target)['last_hidden_state'][:, 0, :]
                scores = []
                for c, s in zip(context, source_section):
                    s = random.choice(all_sections)
                    input = tokenizer([f"{s}{tokenizer.sep_token}{c}"], return_tensors='pt', padding=True, truncation=True, max_length=256).to('cuda' if torch.cuda.is_available() else 'cpu')
                    input_embeddings = model(**input)['last_hidden_state'][:, 0, :]
                    input = torch.cat((source_embeddings, input_embeddings, target_embeddings), dim=1)
                    score = classification_head(input)[0].squeeze().item()
                    scores.append(score)
                position = 1
                for score in scores[1:]:
                    if score > scores[0]:
                        position += 1
                rank.append(position)
        df[column_name] = rank
    
    df.to_parquet('test_ranking_scores.parquet')