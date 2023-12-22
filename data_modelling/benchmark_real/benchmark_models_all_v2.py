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
from glob import glob
from ast import literal_eval

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def fix_title(title):
    return parse.unquote(title).replace('_', ' ')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        required=True, help='Path to data file')
    parser.add_argument('--models_dir', type=str, required=True,
                        help='Directory containing model directories')
    parser.add_argument('--mention_map', type=str,
                        required=True, help='Path to mention map file')
    parser.add_argument('--data_limit', type=int, default=None,
                        help='Limit the number of rows to use')
    parser.add_argument('--column_name', type=str, required=True, help='Name of column to add to dataframe')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--model_name', type=str, required=True, help='Name of model to use')
    parser.add_argument('--loss_function', type=str, required=True, choices=[
                        'ranking', 'indep'], help='Selected loss function used for training')
    parser.add_argument('--use_section_title',
                        action='store_true', help='Use section title in input')
    parser.add_argument('--use_mentions', action='store_true', help='Use mentions in input')
    parser.set_defaults(use_section_title=False, use_mentions=False)

    args = parser.parse_args()

    # check if data path exists
    if not os.path.exists(args.data_path):
        raise ValueError('Data path does not exist')
    # check if models dir exists
    if not os.path.exists(args.models_dir):
        raise ValueError('Models dir does not exist')

    # load model
    dir = os.path.join(args.models_dir, args.model_name)
    model_path = glob(os.path.join(dir, 'model*'))[0]
    classification_head_path = glob(
        os.path.join(dir, 'classification_head*'))[0]
    tokenizer_path = os.path.join(dir, 'tokenizer')
    model = AutoModel.from_pretrained(model_path)
    model.eval()
    model_size = model.config.hidden_size

    if args.loss_function == 'ranking':
        classification_head = nn.Sequential(nn.Linear(model_size, model_size),
                                            nn.ReLU(),
                                            nn.Linear(model_size, 1))
    else:
        classification_head = nn.Sequential(nn.Linear(model_size, model_size),
                                            nn.ReLU(),
                                            nn.Linear(model_size, 2))
    classification_head.load_state_dict(torch.load(
        classification_head_path, map_location=torch.device('cpu')))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    classification_head = classification_head.to(
        'cuda' if torch.cuda.is_available() else 'cpu')

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
    all_sections = set([])
    for i, (pos_context, section) in enumerate(zip(df['context'].tolist(), df['section'].tolist())):
        contexts[i].append(pos_context)
        source_sections[i].append(section)
        all_sections.add(section)
    for i, neg_contexts in enumerate(df['negative_contexts'].tolist()):
        neg_contexts = literal_eval(neg_contexts)
        for context in neg_contexts:
            contexts[i].append(context['context'])
            source_sections[i].append(context['section'])
            all_sections.add(context['section'])
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
    
    for title in mention_map:
        if len(mention_map[title]) > 10:
            mention_map[title] = random.sample(mention_map[title], 10)
            mention_map[title] = ' '.join(mention_map[title])

    print(f'Calculating model rankings. Model type: {args.model_name}')
    rank = []
    with torch.no_grad():
        for context, source_section, source_title, source_lead, target_title, target_lead in tqdm(zip(contexts, source_sections, source_titles, source_leads, target_titles, target_leads), total=len(target_titles)):
            scores = []
            inputs = []
            if target_title not in mention_map:
                mention_map[target_title] = ''
            for c, s in zip(context, source_section):
                input = ["", ""]
                if args.use_mentions:
                    input[0] = f"{target_title} {mention_map[target_title]}{tokenizer.sep_token}{target_lead}"
                else:
                    input[0] = f"{target_title}{tokenizer.sep_token}{target_lead}"
                if args.use_section_title:
                    input[1] = f"{s}{tokenizer.sep_token}"
                input[1] += f"{c}"
                inputs.append(input)
                if len(inputs) == args.batch_size:
                    inputs = tokenizer(inputs, return_tensors='pt', padding='max_length',
                                        truncation=True, max_length=512).to('cuda' if torch.cuda.is_available() else 'cpu')
                    embeddings = model(**inputs)['last_hidden_state'][:, 0, :]
                    if args.loss_function == 'ranking':
                        prediction = classification_head(embeddings).squeeze()
                        if embeddings.shape[0] == 1:
                            scores.append(prediction)
                        else:
                            for score in prediction.tolist():
                                scores.append(score)
                    else:
                        prediction = classification_head(embeddings).squeeze()
                        if embeddings.shape[0] == 1:
                            scores.append(prediction[1])
                        else:
                            for score in prediction.tolist():
                                scores.append(score[1])
                    inputs = []
                    
            if len(inputs) > 0:
                inputs = tokenizer(inputs, return_tensors='pt', padding='max_length',
                                        truncation=True, max_length=512).to('cuda' if torch.cuda.is_available() else 'cpu')
                embeddings = model(**inputs)['last_hidden_state'][:, 0, :]
                if args.loss_function == 'ranking':
                    prediction = classification_head(embeddings).squeeze()
                    if embeddings.shape[0] == 1:
                        scores.append(prediction)
                    else:
                        for score in prediction.tolist():
                            scores.append(score)
                else:
                    prediction = classification_head(embeddings).squeeze()
                    if embeddings.shape[0] == 1:
                        scores.append(prediction[1])
                    else:
                        for score in prediction.tolist():
                            scores.append(score[1])
                inputs = []

            position = 1
            best_score = {'section': source_section[0], 'score': scores[0], 'context': context[0], 'index': 0}
            for i, score in enumerate(scores[1:]):
                if score > scores[0]:
                    position += 1
                    if score > best_score['score']:
                        best_score = {'section': source_section[i+1], 'score': score, 'context': context[i+1], 'index': i+1}
            rank.append(position)

    df[args.column_name] = rank
    df.to_parquet('test_ranking_scores_all.parquet')
