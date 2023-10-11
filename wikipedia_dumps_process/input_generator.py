import argparse
import pandas as pd
import os
from glob import glob
from tqdm import tqdm
import random
import urllib.parse


def unencode_title(title):
    clean_title = urllib.parse.unquote(title).replace('_', ' ')
    return clean_title
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', '-i', type=str,
                        required=True, help='Input directory')
    parser.add_argument('--output_dir', '-o', type=str,
                        required=True, help='Output directory')
    parser.add_argument('--neg_strategies', '-s', type=int, nargs='+', default=[
                        1, 2, 3, 4, 5], help='Negative sampling strategies: 1) replace source with random source not connected to target, 2) replace source with random source connected to target, 3) replace target with random target not connected to source, 4) replace target with random target connected to source, 5) replace context with random context')
    parser.add_argument('--neg_samples_per_pos', '-n', type=int,
                        default=1, help='Number of negative samples per positive sample')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to generate')
    parser.add_argument('--max_val_samples', type=int, default=None, help='Maximum number of validation samples to generate.')
    parser.add_argument('--max_test_samples', type=int, default=None, help='Maximum number of test samples to generate.')
    
    args = parser.parse_args()

    strategies_map = {1: 'easy_replace_source', 2: 'hard_replace_source', 3: 'easy_replace_target',
                      4: 'hard_replace_target', 5: 'replace_context'}
    strategies = []
    for strategy in args.neg_strategies:
        if strategy not in strategies_map:
            raise ValueError(f'Strategy {strategy} not supported')
        strategies.append(strategies_map[strategy])

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not os.path.exists(args.input_dir):
        raise ValueError(f'Input directory {args.input_dir} does not exist')

    print('Loading pages')
    page_files = glob(os.path.join(args.input_dir, 'good_pages*'))
    page_files.sort()
    dfs = []
    for file in tqdm(page_files):
        dfs.append(pd.read_parquet(file, columns=['title', 'lead_paragraph']))
    df_pages = pd.concat(dfs)
    df_pages['title'] = df_pages['title'].apply(unencode_title)
    df_pages = df_pages.to_dict(orient='records')

    print('Loading links')
    link_files = glob(os.path.join(args.input_dir, 'good_links*'))
    link_files.sort()
    dfs = []
    for file in tqdm(link_files):
        dfs.append(pd.read_parquet(file))
    df_links = pd.concat(dfs)
    df_links['source_title'] = df_links['source_title'].apply(unencode_title)
    df_links['target_title'] = df_links['target_title'].apply(unencode_title)
    df_links = df_links.to_dict(orient='records')
    del dfs

    print('Creating auxiliary data structures')
    print('\tProcessing links')
    source_to_all_targets = {}
    target_to_all_sources = {}
    for row in tqdm(df_links):
        source = row['source_title']
        target = row['target_title']
        source_section = row['source_section'].split('<sep>')[0]
        if source not in source_to_all_targets:
            source_to_all_targets[source] = []
        source_to_all_targets[source].append({'target': target, 'section': source_section})
        if target not in target_to_all_sources:
            target_to_all_sources[target] = []
        target_to_all_sources[target].append(source)
        
    print('\tProcessing pages')
    page_leads = {row['title']: row['lead_paragraph'] for row in tqdm(df_pages)}
        
    print('Generating positive samples')
    positive_samples = [{
        'source_title': row['source_title'],
        'source_lead': page_leads[row['source_title']],
        'target_title': row['target_title'],
        'target_lead': page_leads[row['target_title']],
        'link_context': row['context'],
        'source_section': row['source_section'].split('<sep>')[0],
        'label': 1
    } for row in tqdm(df_links)]
    
    print('Generating negative samples')
    negative_samples = []
    for i in tqdm(range(len(positive_samples))):
        valid_strategies = strategies.copy()
        if len(source_to_all_targets[positive_samples[i]['source_title']]) == 1 and 'hard_replace_target' in valid_strategies:
            valid_strategies.remove('hard_replace_target')
        if len(target_to_all_sources[positive_samples[i]['target_title']]) == 1 and 'hard_replace_source' in valid_strategies:
            valid_strategies.remove('hard_replace_source')
        list_strategies = random.choices(valid_strategies, k=args.neg_samples_per_pos)
        new_samples = []
        for strategy in list_strategies:
            new_sample = positive_samples[i].copy()
            if strategy == 'easy_replace_source':
                new_source = random.choices(positive_samples, k=1)[0]['source_title']
                while new_source in target_to_all_sources[positive_samples[i]['target_title']]:
                    new_source = random.choices(positive_samples, k=1)[0]['source_title']
                new_sample['source_title'] = new_source
                new_sample['source_lead'] = page_leads[new_source]
            elif strategy == 'easy_replace_target':
                new_target = random.choices(positive_samples, k=1)[0]['target_title']
                while new_target in source_to_all_targets[positive_samples[i]['source_title']]:
                    new_target = random.choices(positive_samples, k=1)[0]['target_title']
                new_sample['target_title'] = new_target
                new_sample['target_lead'] = page_leads[new_target]
            elif strategy == 'hard_replace_source':
                new_source_section = random.choices(target_to_all_sources[positive_samples[i]['target_title']], k=1)[0]
                new_sample['source_title'] = new_source_section
                new_sample['source_lead'] = page_leads[new_source_section]
            elif strategy == 'hard_replace_target':
                safe_targets = [target['target'] for target in source_to_all_targets[positive_samples[i]['source_title']] if target['section'] != positive_samples[i]['source_section']]
                if len(safe_targets) == 0:
                    new_target = random.choices(positive_samples, k=1)[0]['target_title']
                    while new_target in source_to_all_targets[positive_samples[i]['source_title']]:
                        new_target = new_target = random.choices(positive_samples, k=1)[0]['target_title']
                else:
                    new_target = random.choices(safe_targets, k=1)[0]
                new_sample['target_title'] = new_target
                new_sample['target_lead'] = page_leads[new_target]
            elif strategy == 'replace_context':
                new_context = random.choices(positive_samples, k=1)[0]['link_context']
                while new_context == positive_samples[i]['link_context']:
                    new_context = random.choices(positive_samples, k=1)[0]['link_context']
                new_sample['link_context'] = new_context
            new_sample['label'] = 0
            new_samples.append(new_sample)
        negative_samples.extend(new_samples)   
    
    print('Spliting data into train, test, val')
    df = pd.DataFrame(positive_samples + negative_samples)
    if args.max_samples:
        df = df.sample(min(args.max_samples, len(df))).reset_index(drop=True)

    if args.max_val_samples and not args.max_test_samples:
        args.max_test_samples = args.max_val_samples
    if args.max_test_samples and not args.max_val_samples:
        args.max_val_samples = args.max_test_samples
    
    if args.max_val_samples:
        train_samples = len(df) - args.max_val_samples - args.max_test_samples
        df_train = df.sample(train_samples)
        df_val = df.drop(df_train.index).sample(args.max_val_samples)
        df_test = df.drop(df_train.index).drop(df_val.index).sample(args.max_test_samples)
    else:
        df_train = df.sample(frac=0.8)
        df_val = df.drop(df_train.index).sample(frac=0.5)
        df_test = df.drop(df_train.index).drop(df_val.index).sample(frac=1.0)

    print('Saving data')
    df_train.reset_index(drop=True).to_parquet(os.path.join(args.output_dir, 'train', 'train.parquet'))
    df_val.reset_index(drop=True).to_parquet(os.path.join(args.output_dir, 'val', 'val.parquet'))
    df_test.reset_index(drop=True).to_parquet(os.path.join(args.output_dir, 'test', 'test.parquet')) 