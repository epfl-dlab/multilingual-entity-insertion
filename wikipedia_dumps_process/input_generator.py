import argparse
import pandas as pd
import os
from glob import glob
from tqdm import tqdm
import random
import urllib.parse
from nltk import sent_tokenize
import re


def unencode_title(title):
    clean_title = urllib.parse.unquote(title).replace('_', ' ')
    return clean_title


def get_strategies(link, strategies, k):
    valid_strategies = strategies.copy()
    if len(source_to_all_targets[link['source_title']]) == 1 and 'hard_replace_target' in valid_strategies:
        valid_strategies.remove('hard_replace_target')
    if len(target_to_all_sources[link['target_title']]) == 1 and 'hard_replace_source' in valid_strategies:
        valid_strategies.remove('hard_replace_source')
    list_strategies = random.choices(
        valid_strategies, k=k)
    return list_strategies


def easy_replace_source(link, all_links):
    new_source = random.choices(all_links, k=1)[
        0]['source_title']
    while new_source in target_to_all_sources[link['target_title']]:
        new_source = random.choices(all_links, k=1)[
            0]['source_title']
    link['source_title'] = new_source
    link['source_lead'] = page_leads[new_source]
    link['neg_type'] = 'easy_replace_source'
    return link


def easy_replace_target(link, all_links):
    new_target = random.choices(all_links, k=1)[
        0]['target_title']
    while new_target in source_to_all_targets[sample['source_title']]:
        new_target = random.choices(all_links, k=1)[
            0]['target_title']
    link['target_title'] = new_target
    link['target_lead'] = page_leads[new_target]
    link['neg_type'] = 'easy_replace_target'
    return link


def hard_replace_source(link):
    new_source_section = random.choices(
        target_to_all_sources[link['target_title']], k=1)[0]
    link['source_title'] = new_source_section
    link['source_lead'] = page_leads[new_source_section]
    link['neg_type'] = 'hard_replace_source'
    return link


def hard_replace_target(link, all_links):
    safe_targets = []
    for target in source_to_all_targets[link['source_title']]:
        found = False
        for mention in entity_map[target]:
            if mention in link['link_context']:
                found = True
                break
        if not found:
            safe_targets.append(target)
    if len(safe_targets) == 0:
        link = easy_replace_target(link, all_links)
        link['neg_type'] = 'easy_replace_target'
    else:
        new_target = random.choices(safe_targets, k=1)[0]
        link['neg_type'] = 'hard_replace_target'
    link['target_title'] = new_target
    link['target_lead'] = page_leads[new_target]
    return link


def easy_replace_context(link, page_sections):
    while True:
        new_file = random.choices(page_sections.keys(), k=1)[0]
        while new_file == link['source_title']:
            new_file = random.choices(page_sections.keys(), k=1)[0]
        new_section = random.choices(page_sections[new_file].keys(), k=1)[0]
        middle_sentence_index = random.randint(
            0, len(page_sections[new_file][new_section]) - 1)
        new_context = " ".join(page_sections[new_file][new_section][max(
            0, middle_sentence_index - 5):min(middle_sentence_index + 6, len(page_sections[new_file][new_section]))])

        found = False
        for mention in entity_map[link['target_title']]:
            if mention in new_context:
                found = True
                break
        if not found:
            break
    link['link_context'] = new_context
    link['neg_type'] = 'easy_replace_context'
    return link


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir_train', type=str,
                        required=True, help='Input directory')
    parser.add_argument('--input_month2_dir', type=str,
                        required=True, help='Input directory for second month data')
    parser.add_argument('--input_dir_val', type=str,
                        required=True, help='Input directory')
    parser.add_argument('--output_dir', type=str,
                        required=True, help='Output directory')
    parser.add_argument('--neg_strategies', '-s', type=int, nargs='+', default=[
                        1, 2, 3, 4, 5, 6], help='Negative sampling strategies: 1) replace source with random source not connected to target, 2) replace source with random source connected to target, 3) replace target with random target not connected to source, 4) replace target with random target connected to source, 5) replace context with random context, 6) replace context with context from the same source page')
    parser.add_argument('--neg_samples_train', type=int,
                        default=1, help='Number of negative samples per positive sample for training')
    parser.add_argument('--neg_samples_val', type=int,
                        default=1, help='Number of negative samples per positive sample for validation')
    parser.add_argument('--max_train_samples', type=int, default=None,
                        help='Maximum number of train samples to generate')
    parser.add_argument('--max_val_samples', type=int, default=None,
                        help='Maximum number of validation samples to generate.')
    parser.add_argument('--max_test_samples', type=int, default=None,
                        help='Maximum number of test samples to generate.')

    args = parser.parse_args()

    strategies_map = {1: 'easy_replace_source', 2: 'hard_replace_source', 3: 'easy_replace_target',
                      4: 'hard_replace_target', 5: 'easy_replace_context', 6: 'hard_replace_context'}
    strategies = []
    for strategy in args.neg_strategies:
        if strategy not in strategies_map:
            raise ValueError(f'Strategy {strategy} not supported')
        strategies.append(strategies_map[strategy])

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not os.path.exists(args.input_dir_train):
        raise ValueError(
            f'Train input directory {args.input_dir_train} does not exist')
    if not os.path.exists(args.input_dir_val):
        raise ValueError(
            f'Validation input directory {args.input_dir_val} does not exist')

    print('Loading pages')
    page_files = glob(os.path.join(args.input_dir_train, 'good_pages*')) + \
        glob(os.path.join(args.input_month2_dir, 'good_pages*'))
    page_files.sort()
    dfs = []
    for file in tqdm(page_files):
        dfs.append(pd.read_parquet(file, columns=['title', 'lead_paragraph']))
    df_pages = pd.concat(dfs)
    df_pages = df_pages.drop_duplicates(
        subset=['title']).reset_index(drop=True)
    df_pages['title'] = df_pages['title'].apply(unencode_title)
    df_pages = df_pages.to_dict(orient='records')

    print('Loading training links')
    link_files = glob(os.path.join(args.input_dir_train, 'good_links*'))
    link_files.sort()
    dfs = []
    for file in tqdm(link_files):
        dfs.append(pd.read_parquet(file))
    df_links_train = pd.concat(dfs)
    df_links_train['source_title'] = df_links_train['source_title'].apply(
        unencode_title)
    df_links_train['target_title'] = df_links_train['target_title'].apply(
        unencode_title)
    df_links_train = df_links_train.to_dict(orient='records')

    print('Loading training section texts')
    section_files = glob(os.path.join(args.input_dir_train, 'sections*'))
    section_files.sort()
    dfs = []
    for file in tqdm(section_files):
        dfs.append(pd.read_parquet(file))
    df_sections_train = pd.concat(dfs)
    df_sections_train['title'] = df_sections_train['title'].apply(
        unencode_title)
    df_sections_train = df_sections_train.to_dict(orient='records')

    print('Loading validation links')
    df_links_val = pd.read_parquet(os.path.join(
        args.input_dir_val, 'val_links.parquet'))
    df_links_val['source_title'] = df_links_val['source_title'].apply(
        unencode_title)
    df_links_val['target_title'] = df_links_val['target_title'].apply(
        unencode_title)
    df_links_val = df_links_val.to_dict(orient='records')

    print('Loading validation section texts')
    section_files = glob(os.path.join(args.input_month2_dir, 'sections*'))
    section_files.sort()
    dfs = []
    for file in tqdm(section_files):
        dfs.append(pd.read_parquet(file))
    df_sections_val = pd.concat(dfs)
    df_sections_val['title'] = df_sections_val['title'].apply(
        unencode_title)
    df_sections_val = df_sections_val.to_dict(orient='records')

    print('Loading mention map')
    mention_map_1 = pd.read_parquet(os.path.join(
        args.input_dir_train, 'mention_map.parquet'))
    mention_map_1 = mention_map_1.to_dict(orient='records')
    mention_map_2 = pd.read_parquet(os.path.join(
        args.input_month2_dir, 'mention_map.parquet'))
    mention_map_2 = mention_map_2.to_dict(orient='records')
    entity_map = {}
    for row in mention_map_1:
        title = unencode_title(row['target_title'])
        mention = row['mention']
        if title in entity_map:
            entity_map[title].add(mention)
        else:
            entity_map[title] = set([mention])
    for row in mention_map_2:
        title = unencode_title(row['target_title'])
        mention = row['mention']
        if title in entity_map:
            entity_map[title].add(mention)
        else:
            entity_map[title] = set([mention])

    print('Creating auxiliary data structures')
    print('\tProcessing links')
    source_to_all_targets = {}
    target_to_all_sources = {}
    for row in tqdm(df_links_train + df_links_val):
        source = row['source_title']
        target = row['target_title']
        source_section = row['source_section'].split('<sep>')[0]
        if source not in source_to_all_targets:
            source_to_all_targets[source] = []
        source_to_all_targets[source].append(target)
        if target not in target_to_all_sources:
            target_to_all_sources[target] = []
        target_to_all_sources[target].append(source)

    print('\tProcessing pages')
    page_leads = {row['title']: row['lead_paragraph']
                  for row in tqdm(df_pages)}

    print('\tProcessing sections')
    page_sections_train = {}
    for row in tqdm(df_sections_train):
        title = row['title']
        text = row['text']
        section = row['section'].split('<sep>')[0]
        if title not in page_sections_train:
            page_sections_train[title] = {}
        if section not in page_sections_train[title]:
            page_sections_train[title][section] = []
        text = text.strip()
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r' +', ' ', text)
        sentences = [elem.strip() + '\n' for elem in text.split('\n')
                     if elem.strip() != '']
        if sentences:
            sentences[-1] = sentences[-1][:-1]
        for sentence in sentences:
            for s in sent_tokenize(sentence):
                s = s.strip()
                s = re.sub(' +', ' ', s)
                s = re.sub('\n+', '\n', s)
                s = re.sub('\n +', '\n', s)
                if s:
                    page_sections_train[title][section].append(s)
    page_sections_val = {}
    for row in tqdm(df_sections_val):
        title = row['title']
        text = row['text']
        if title not in page_sections_val:
            page_sections_val[title] = []
        text = text.strip()
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r' +', ' ', text)
        sentences = [elem.strip() + '\n' for elem in text.split('\n')
                     if elem.strip() != '']
        if sentences:
            sentences[-1] = sentences[-1][:-1]
        for sentence in sentences:
            for s in sent_tokenize(sentence):
                s = s.strip()
                s = re.sub(' +', ' ', s)
                s = re.sub('\n+', '\n', s)
                s = re.sub('\n +', '\n', s)
                if s:
                    page_sections_val[title].append(s)

    print('Generating positive samples')
    positive_samples_train = [{
        'source_title': row['source_title'],
        'source_lead': page_leads[row['source_title']],
        'target_title': row['target_title'],
        'target_lead': page_leads[row['target_title']],
        'link_context': row['context'],
        'source_section': row['source_section'].split('<sep>')[0],
        'context_span_start_index': row['context_span_start_index'],
        'context_span_end_index': row['context_span_end_index'],
        'context_sentence_start_index': row['context_sentence_start_index'],
        'context_sentence_end_index': row['context_sentence_end_index'],
        'context_mention_start_index': row['context_mention_start_index'],
        'context_mention_end_index': row['context_mention_end_index'],
        'label': 1,
        'neg_type': 'none'
    } for row in tqdm(df_links_train)]

    positive_samples_val = []
    for row in tqdm(df_links_val):
        try:
            positive_samples_val.append({
                'source_title': row['source_title'],
                'source_lead': page_leads[row['source_title']],
                'target_title': row['target_title'],
                'target_lead': page_leads[row['target_title']],
                'link_context': row['context'],
                'source_section': row['source_section'].split('<sep>')[0],
                'context_span_start_index': row['context_span_start_index'],
                'context_span_end_index': row['context_span_end_index'],
                'context_sentence_start_index': row['context_sentence_start_index'],
                'context_sentence_end_index': row['context_sentence_end_index'],
                'context_mention_start_index': row['context_mention_start_index'],
                'context_mention_end_index': row['context_mention_end_index'],
                'label': 1,
                'neg_type': 'none',
                'noise_strategy': row['noise_strategy']
            })
        except:
            print(
                f"Couldn't find {row['target_title']} from {row['source_title']}")

    print('Generating negative samples')
    negative_samples_train = []
    for sample in tqdm(positive_samples_train):
        list_strategies = get_strategies(
            sample, strategies, args.neg_samples_train)
        new_samples = []
        for strategy in list_strategies:
            new_sample = sample.copy()
            if strategy == 'easy_replace_source':
                new_sample = easy_replace_source(
                    new_sample, positive_samples_train)
            elif strategy == 'easy_replace_target':
                new_sample = easy_replace_target(
                    new_sample, positive_samples_train)
            elif strategy == 'hard_replace_source':
                new_sample = hard_replace_source(new_sample)
            elif strategy == 'hard_replace_target':
                new_sample = hard_replace_target(
                    new_sample, positive_samples_train)
            elif strategy == 'easy_replace_context':
                new_sample = easy_replace_context(
                    new_sample, page_sections_train)
            elif strategy == 'hard_replace_context':
                new_sample = hard_replace_context(
                    new_sample, page_sections_train)

            new_sample['label'] = 0
            new_samples.append(new_sample)
        negative_samples_train.extend(new_samples)

    negative_samples_val = []
    for sample in tqdm(positive_samples_val):
        list_strategies = get_strategies(
            sample, strategies, args.neg_samples_val)
        new_samples = []
        for strategy in list_strategies:
            new_sample = sample.copy()
            if strategy == 'easy_replace_source':
                new_sample = easy_replace_source(
                    new_sample, positive_samples_val)
            elif strategy == 'easy_replace_target':
                new_sample = easy_replace_target(
                    new_sample, positive_samples_val)
            elif strategy == 'hard_replace_source':
                new_sample = hard_replace_source(new_sample)
            elif strategy == 'hard_replace_target':
                new_sample = hard_replace_target(
                    new_sample, positive_samples_val)
            elif strategy == 'easy_replace_context':
                new_sample = easy_replace_context(
                    new_sample, page_sections_val)
            elif strategy == 'hard_replace_context':
                new_sample = hard_replace_context(
                    new_sample, page_sections_val)
            new_sample['label'] = 0
            new_samples.append(new_sample)
        negative_samples_val.extend(new_samples)

    print('Saving data')
    df_train = pd.DataFrame(positive_samples_train + negative_samples_train)
    df_train = df_train.sample(
        n=min(args.max_train_samples, len(df_train))).reset_index(drop=True)

    df_val_full = pd.DataFrame(
        positive_samples_val + negative_samples_val).sample(frac=1).reset_index(drop=True)
    if args.max_val_samples and not args.max_test_samples:
        args.max_test_samples = len(df_val_full) - args.max_val_samples
    if args.max_test_samples and not args.max_val_samples:
        args.max_val_samples = len(df_val_full) - args.max_test_samples

    if args.max_val_samples and args.max_test_samples:
        if args.max_val_samples + args.max_test_samples > len(df_val_full):
            # keep the same ratio
            val_samples = int(len(df_val_full) * args.max_val_samples /
                              (args.max_val_samples + args.max_test_samples))
            test_samples = len(df_val_full) - val_samples
        else:
            val_samples = args.max_val_samples
            test_samples = args.max_test_samples
    elif args.max_val_samples:
        val_samples = args.max_val_samples
        test_samples = len(df_val_full) - val_samples
    elif args.max_test_samples:
        test_samples = args.max_test_samples
        val_samples = len(df_val_full) - test_samples
    else:
        val_samples = int(len(df_val_full) * 0.5)
        test_samples = len(df_val_full) - val_samples

    df_val = df_val_full.sample(n=val_samples)
    df_test = df_val_full.drop(df_val.index).sample(n=test_samples)

    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    # create directories if they don't exist
    if not os.path.exists(os.path.join(args.output_dir, 'train')):
        os.makedirs(os.path.join(args.output_dir, 'train'))
    if not os.path.exists(os.path.join(args.output_dir, 'val')):
        os.makedirs(os.path.join(args.output_dir, 'val'))
    if not os.path.exists(os.path.join(args.output_dir, 'test')):
        os.makedirs(os.path.join(args.output_dir, 'test'))
    df_train.to_parquet(os.path.join(
        args.output_dir, 'train', 'train.parquet'))
    df_val.to_parquet(os.path.join(args.output_dir, 'val', 'val.parquet'))
    df_test.to_parquet(os.path.join(args.output_dir, 'test', 'test.parquet'))
