import argparse
import pandas as pd
import sys
import os
from urllib import parse
from tqdm import tqdm
import random
from ast import literal_eval
import torch
from transformers import BertTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from baselines import bm25
from baselines import exact_match
from baselines import fuzzy_match
from baselines import embedding_similarity
from baselines import entqa
from baselines import get

def fix_title(title):
    return parse.unquote(title).replace('_', ' ')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the directory containing the data')
    parser.add_argument('--langs', type=str, nargs='+', required=True, help='Languages to benchmark')
    parser.add_argument('--method_name', type=str, required=True, choices=[
                        'embedding_similarity', 'random', 'bm25', 'bm25_mentions', 'exact_match', 'fuzzy_match', 'entqa', 'get', 'all'], help='Which method to use')
    parser.add_argument('--model_name', type=str, default='BAAI/bge-base-en-v1.5', help='Which model to use')
    
    args = parser.parse_args()

    # check if data path exists
    if not os.path.exists(args.data_dir):
        raise ValueError('Data dir does not exist')
    
    if args.method_name == 'all':
        args.method_name = ['random', 'bm25', 'bm25_mentions', 'exact_match', 'fuzzy_match',]
    else:
        args.method_name = [args.method_name]
    
    # process each language sequentially
    for lang in args.langs:
        print(f'Processing language {lang}')
        try:
            df = pd.read_parquet(os.path.join(args.data_dir, f'{lang}.parquet'))
        except:
            print(f'No data for language {lang}')
            continue
        
        source_titles = df['source_title'].apply(fix_title).tolist()
        source_leads = df['source_lead'].tolist()
        target_titles = df['target_title'].apply(fix_title).tolist()
        target_leads = df['target_lead'].tolist()
        
        contexts = [[] for _ in range(len(df))]
        source_sections = [[] for _ in range(len(df))]
        for i, (pos_context, section) in enumerate(zip(df['context'].tolist(), df['section'].tolist())):
            contexts[i].append(pos_context)
            source_sections[i].append(section)
        for i, neg_contexts in enumerate(df['negative_contexts'].tolist()):
            neg_contexts = literal_eval(neg_contexts)
            for context in neg_contexts:
                contexts[i].append(context['context'])
                source_sections[i].append(context['section'])

        mention_map = pd.read_parquet(os.path.join(args.data_dir, f'{lang}_mention_map.parquet'))
        mention_map_pre = mention_map.to_dict('records')
        mention_map = {}
        for row in mention_map_pre:
            title = fix_title(row['target_title'])
            if title in mention_map:
                mention_map[title].append(row['mention'])
            else:
                mention_map[title] = [row['mention']]
        
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
                scores = bm25.rank_contexts(context, title, lead, use_stopwords=lang in ['en', 'simple'], use_japanese=lang=='ja')
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
                scores = bm25.rank_contexts(context, title, lead, mention_map[title], use_stopwords=lang in ['en', 'simple'], use_japanese=lang=='ja')
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
                scores = fuzzy_match.rank_contexts(context, mention_map[title], use_stopwords=lang in ['en', 'simple'], use_japanese=lang=='ja')
                position = 1
                equals = 0
                for score in scores[1:]:
                    if score > scores[0]:
                        position += 1
                    if score == scores[0]:
                        equals += 1
                rank.append(position + random.randint(0, equals))
            df['fuzzy_match_rank'] = rank
        
        if 'entqa' in args.method_name:
            print('Calculating EntQA ranks')
            model = entqa.load_model(True, '../baselines/EntQA/models/biencoder_wiki_large.json', '../baselines/EntQA/retriever.pt', torch.device('cuda'), None, True)
            tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
            model = model.to('cuda')
            model = model.eval()
            rank = []
            mrr = 0
            mrr_random = 0
            counter = 0
            for context, title, lead in tqdm(zip(contexts, target_titles, target_leads), total=len(target_titles)):
                scores = entqa.rank_contexts(context, title, lead, model, tokenizer)
                position = 1
                equals = 0
                for score in scores[1:]:
                    if score > scores[0]:
                        position += 1
                    if score == scores[0]:
                        equals += 1
                rank.append(position + random.randint(0, equals))
                mrr = mrr * counter / (counter + 1) + 1 / rank[-1] / (counter + 1)
                mrr_random = mrr_random * counter / (counter + 1) + 1 / random.randint(1, len(scores)) / (counter + 1)
                counter += 1
                print(rank[-1], len(scores), mrr, mrr_random)
            df['entqa_rank'] = rank
        
        if 'get' in args.method_name:
            print('Calculating GET ranks')
            model = AutoModelForSeq2SeqLM.from_pretrained('../baselines/GROOV/model_checkpoint/model')
            tokenizer = AutoTokenizer.from_pretrained('t5-base', model_max_length=512)
            
            model.to('cuda')
            model.eval()
            rank = []
            mrr = 0
            mrr_random = 0
            counter = 0
            for context, title in tqdm(zip(contexts, target_titles), total=len(target_titles)):
                scores = get.rank_contexts(context, title, model, tokenizer)
                position = 1
                equals = 0
                for score in scores[1:]:
                    if score > scores[0]:
                        position += 1
                    if score == scores[0]:
                        equals += 1
                rank.append(position + random.randint(0, equals))
                mrr = mrr * counter / (counter + 1) + 1 / rank[-1] / (counter + 1)
                mrr_random = mrr_random * counter / (counter + 1) + 1 / random.randint(1, len(scores)) / (counter + 1)
                counter += 1
                print(rank[-1], len(scores), mrr, mrr_random)
            df['get_rank'] = rank

        df.to_parquet(os.path.join(args.data_dir, f'{lang}.parquet'))            