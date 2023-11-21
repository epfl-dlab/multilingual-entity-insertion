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
    parser.add_argument('--model_name', type=str, default='', help='Name of model to use')
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
    parser.add_argument('--mask_negatives', help='Mask negative examples', action='store_true')
    parser.set_defaults(use_corruption=False, use_section_title=False,
                        use_section_title_random=False, use_mentions=False, mask_negatives=False)

    args = parser.parse_args()

    # check if data path exists
    if not os.path.exists(args.data_path):
        raise ValueError('Data path does not exist')
    # check if mention map exists
    if not os.path.exists(args.mention_map):
        raise ValueError('Mention map does not exist')
    # check if models dir exists
    if not os.path.exists(args.models_dir):
        raise ValueError('Models dir does not exist')

    # find model name
    if args.model_name == '':
        model_name = args.loss_function
        if args.use_corruption:
            model_name += '_corrupt'
        if args.use_section_title or args.use_section_title_random:
            model_name += '_section'
        if args.use_mentions:
            model_name += '_mentions'
        if args.mask_negatives:
            model_name += '_negmask'
    else:
        model_name = args.model_name

    # load model
    dir = os.path.join(args.models_dir, model_name)
    model_path = glob(os.path.join(dir, 'model*'))[0]
    classification_head_path = glob(
        os.path.join(dir, 'classification_head*'))[0]
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

    print(f'Calculating model rankings. Model type: {model_name}')
    rank = []
    with torch.no_grad():
        for context, source_section, source_title, source_lead, target_title, target_lead in tqdm(zip(contexts, source_sections, source_titles, source_leads, target_titles, target_leads), total=len(target_titles)):
            if target_title not in mention_map:
                mention_map[target_title] = [target_title]
            source = tokenizer([f"{source_title}{tokenizer.sep_token}{source_lead}"], return_tensors='pt',
                                 padding=True, truncation=True, max_length=256).to('cuda' if torch.cuda.is_available() else 'cpu')
            target = tokenizer([f"{target_title}{tokenizer.sep_token}{target_lead}"], return_tensors='pt',
                                    padding=True, truncation=True, max_length=256).to('cuda' if torch.cuda.is_available() else 'cpu')
            source_embeddings = model(
                **source)['last_hidden_state'][:, 0, :]
            target_embeddings = model(
                **target)['last_hidden_state'][:, 0, :]
            source_embeddings = source_embeddings.expand(args.batch_size, -1)
            target_embeddings = target_embeddings.expand(args.batch_size, -1)
            scores = []
            inputs = []
            for c, s in zip(context, source_section):
                input = ''
                if args.use_section_title or args.use_section_title_random:
                    if args.use_section_title_random:
                        s = random.choice(all_sections)
                    input += f"{s}{tokenizer.sep_token}"
                if args.use_mentions:
                    input += f"{' '.join(mention_map[target_title])}{tokenizer.sep_token}"
                input += f"{c}"
                inputs.append(input)
                if len(inputs) == args.batch_size:
                    inputs = tokenizer(inputs, return_tensors='pt', padding=True,
                                        truncation=True, max_length=256).to('cuda' if torch.cuda.is_available() else 'cpu')
                    input_embeddings = model(
                        **inputs)['last_hidden_state'][:, 0, :]
                    input = torch.cat(
                        (source_embeddings, input_embeddings, target_embeddings), dim=1)
                    if args.loss_function == 'ranking':
                        prediction = classification_head(input).squeeze()
                        if input.shape[0] == 1:
                            scores.append(prediction)
                        else:
                            for score in prediction.tolist():
                                scores.append(score)
                    else:
                        prediction = classification_head(input).squeeze()
                        if input.shape[0] == 1:
                            scores.append(prediction[1])
                        else:
                            for score in prediction.tolist():
                                scores.append(score[1])
                    inputs = []
            if len(inputs) > 0:
                inputs = tokenizer(inputs, return_tensors='pt', padding=True,
                                    truncation=True, max_length=256).to('cuda' if torch.cuda.is_available() else 'cpu')
                input_embeddings = model(
                    **inputs)['last_hidden_state'][:, 0, :]
                source_embeddings = source_embeddings[:input_embeddings.shape[0], :]
                target_embeddings = target_embeddings[:input_embeddings.shape[0], :]
                input = torch.cat(
                    (source_embeddings, input_embeddings, target_embeddings), dim=1)
                if args.loss_function == 'ranking':
                    prediction = classification_head(input).squeeze()
                    if input.shape[0] == 1:
                        scores.append(prediction)
                    else:
                        for score in prediction.tolist():
                            scores.append(score)
                else:
                    prediction = classification_head(input).squeeze()
                    if input.shape[0] == 1:
                        scores.append(prediction[1])
                    else:
                        for score in prediction.tolist():
                            scores.append(score[1])
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
