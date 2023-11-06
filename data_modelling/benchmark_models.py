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

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

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
    parser.add_argument('--loss_function', type=str, required=True, choices=[
                        'ranking', 'indep'], help='Selected loss function used for training')
    parser.add_argument('--use_corruption', action='store_true',
                        help='Use model trained with corruption')
    parser.add_argument('--use_section_title',
                        action='store_true', help='Use section title in input')
    parser.add_argument('--use_section_title_random',
                        action='store_true', help='Use random section title in input')
    parser.add_argument('--use_mentions', action='store_true',
                        help='Use mentions in input')
    parser.set_defaults(use_corruption=False, use_section_title=False,
                        use_section_title_random=False, use_mentions=False)

    args = parser.parse_args()

    # check if data path exists
    if not os.path.exists(args.data_path):
        raise ValueError('Data path does not exist')
    # check if mention map exists
    if not os.path.exists(args.mention_map):
        raise ValueError('Mention map does not exist')

    # find model name
    model_name = args.loss_function
    if args.use_corruption:
        model_name += '_corrupt'
    if args.use_section_title or args.use_section_title_random:
        model_name += '_section'
    if args.use_mentions:
        model_name += '_mentions'

    # load model
    dir = os.path.join(args.models_dir, model_name)
    model_path = glob(os.path.join(dir, 'model_*'))[0]
    classification_head_path = glob(
        os.path.join(dir, 'classification_head_*'))[0]
    tokenizer_path = os.path.join(dir, 'tokenizer')

    model = AutoModel.from_pretrained(model_path)
    model.eval()
    if args.loss_function == 'ranking':
        classification_head = nn.Sequential(nn.Linear(model.config.hidden_size * 3, model.config.hidden_size),
                                            nn.ReLU(),
                                            nn.Linear(model.config.hidden_size, 1))
    else:
        classification_head = nn.Sequential(nn.Linear(model.config.hidden_size * 3, model.config.hidden_size),
                                            nn.ReLU(),
                                            nn.Linear(model.config.hidden_size, 2))
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

    print(f'Calculating model rankings. Model type: {model_name}')
    rank = []
    with torch.no_grad():
        for context, source_section, source_title, source_lead, target_title, target_lead in tqdm(zip(contexts, source_sections, source_titles, source_leads, target_titles, target_leads), total=len(target_titles)):
            if target_title not in mention_map:
                mention_map[target_title] = []
            source = tokenizer([f"{source_title}{tokenizer.sep_token}{source_lead}"], return_tensors='pt',
                                 padding=True, truncation=True, max_length=256).to('cuda' if torch.cuda.is_available() else 'cpu')
            target = tokenizer([f"{target_title}{tokenizer.sep_token}{target_lead}"], return_tensors='pt',
                                    padding=True, truncation=True, max_length=256).to('cuda' if torch.cuda.is_available() else 'cpu')
            source_embeddings = model(
                **source)['last_hidden_state'][:, 0, :]
            target_embeddings = model(
                **target)['last_hidden_state'][:, 0, :]
            scores = []
            for c, s in zip(context, source_section):
                input = ''
                if args.use_section_title or args.use_section_title_random:
                    if args.use_section_title_random:
                        s = random.choice(all_sections)
                    input += f"{s}{tokenizer.sep_token}"
                if args.use_mentions:
                    input += f"{' '.join(mention_map[target_title])}{tokenizer.sep_token}"
                input += f"{c}"
                input = tokenizer([input], return_tensors='pt', padding=True,
                                    truncation=True, max_length=256).to('cuda' if torch.cuda.is_available() else 'cpu')
                input_embeddings = model(
                    **input)['last_hidden_state'][:, 0, :]
                input = torch.cat(
                    (source_embeddings, input_embeddings, target_embeddings), dim=1)
                if args.loss_function == 'ranking':
                    score = classification_head(input)[0].squeeze().item()
                else:
                    score = classification_head(input)[0].squeeze().tolist()[1]
                scores.append(score)
            position = 1
            for score in scores[1:]:
                if score > scores[0]:
                    position += 1
            rank.append(position)
        
    df[args.column_name] = rank
    df.to_parquet('test_ranking_scores.parquet')
