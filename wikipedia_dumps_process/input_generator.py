import argparse
from calendar import c
import pandas as pd
import os
from glob import glob
from tqdm import tqdm
import random
import urllib.parse
from nltk import sent_tokenize
import re
from multiprocessing import Pool, cpu_count
import math


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


def construct_negative_samples(positive_samples, neg_samples, page_sections):
    pages = list(page_sections.keys())
    negative_samples = []
    for sample in tqdm(positive_samples):
        list_strategies = get_strategies(
            sample, strategies, neg_samples)
        new_samples = []
        available_sentences = {}
        for key in page_sections[sample['source_title']]:
            available_sentences[key] = [i for i in range(len(page_sections[sample['source_title']][key]['sentences']))]
        for strategy in list_strategies:
            new_sample = sample.copy()
            new_sample['label'] = 0
            if strategy == 'easy_replace_source':
                new_sample = easy_replace_source(
                    new_sample, positive_samples)
            elif strategy == 'easy_replace_target':
                new_sample = easy_replace_target(
                    new_sample, positive_samples)
            elif strategy == 'hard_replace_source':
                new_sample = hard_replace_source(new_sample)
            elif strategy == 'hard_replace_target':
                new_sample = hard_replace_target(
                    new_sample, positive_samples)
            elif strategy == 'easy_replace_context':
                new_sample = easy_replace_context(
                    new_sample, page_sections, pages)
                while new_sample in new_samples:
                    new_sample = easy_replace_context(
                        new_sample, page_sections, pages)    
            elif strategy == 'hard_replace_context':
                new_sample = hard_replace_context(
                    new_sample, page_sections, pages, available_sentences)
                while new_sample in new_samples:
                    new_sample = easy_replace_context(
                        new_sample, page_sections, pages)
            new_samples.append(new_sample)
        negative_samples.extend(new_samples)
    return negative_samples


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


def easy_replace_context(link, page_sections, pages):
    while True:
        new_file = random.choices(pages, k=1)[0]
        if new_file == link['source_title']:
            continue
        new_context = replace_context(new_file, link['target_title'], int(
            link['depth'].split('.')[0]), page_sections)
        if new_context is not None:
            break
    link['link_context'] = new_context['context']
    link['source_section'] = new_context['section']
    link['neg_type'] = 'easy_replace_context'
    return link


def hard_replace_context(link, page_sections, pages, available_sentences):
    new_context = replace_context(link['source_title'], link['target_title'], int(
        link['depth'].split('.')[0]), page_sections, available_sentences)
    if new_context is None:
        link = easy_replace_context(link, page_sections, pages)
        link['neg_type'] = 'easy_replace_context'
    else:
        link['link_context'] = new_context['context']
        link['source_section'] = new_context['section']
        link['neg_type'] = 'hard_replace_context'
    return link


def replace_context(source_title, target_title, source_depth, page_sections, available_sentences=None):
    valid_section_ranges = {}
    for key in page_sections[source_title]:
        # don't sample from the same section
        if page_sections[source_title][key]['depth'] == source_depth:
            continue
        # if there are no more available sentences in this section, skip it
        if available_sentences is not None and len(available_sentences[key]) == 0:
            continue
        good_sentences = []
        for i, sentence in enumerate(page_sections[source_title][key]['sentences']):
            found = False
            for mention in entity_map[target_title]:
                if mention in sentence and mention != '':
                    found = True
                    break
            if not found:
                good_sentences.append(i)
        if not good_sentences:
            continue
        # find all the consecutive ranges in the good sentences
        # store them as a list of tuples (start, end)
        ranges = []
        if len(good_sentences) > 0:
            start = good_sentences[0]
            end = good_sentences[0]
            for i in range(1, len(good_sentences)):
                if good_sentences[i] == end + 1:
                    end = good_sentences[i]
                else:
                    if end > start:
                        ranges.append((start, end))
                    start = good_sentences[i]
                    end = good_sentences[i]
            if end > start:
                ranges.append((start, end))
        if ranges:
            data = {'depth': page_sections[source_title][key]['depth'], 'ranges': []}
            if available_sentences is not None:
                for r in ranges:
                    # find if range contains any of the available sentences
                    range_available = []
                    for index in available_sentences[key]:
                        if index >= r[0] and index <= r[1]:
                            range_available.append(index)
                    if len(range_available) > 0:
                        data['ranges'].append({'range': r, 'available': range_available})
            else:
                data['ranges'] = [{'range': r, 'available': [i for i in range(r[0], r[1] + 1)]} for r in ranges]
            if len(data['ranges']) > 0:
                valid_section_ranges[key] = data

    if len(valid_section_ranges) == 0:
        return None
    else:
        keys = list(valid_section_ranges.keys())
        weights = [1 / abs(valid_section_ranges[key]
                           ['depth'] - source_depth) for key in keys]
        new_section = random.choices(keys, weights=weights, k=1)[0]
        new_sentence_range = random.choices(
            valid_section_ranges[new_section]['ranges'], k=1)[0]
        new_sentence_index = random.choices(new_sentence_range['available'], k=1)[0]
        left_limit = max(new_sentence_range['range'][0], new_sentence_index - 5)
        right_limit = min(new_sentence_index + 6, new_sentence_range['range'][1] + 1)
        new_context = " ".join(page_sections[source_title][new_section]['sentences'][left_limit:right_limit])
        # remove all the sentences from the available indices that would produce the same context
        if available_sentences:
            if left_limit == new_sentence_range['range'][0] and right_limit == new_sentence_range['range'][1] + 1:
                for i in range(new_sentence_range['range'][1] - 5, new_sentence_range['range'][0] + 6):
                    if i in available_sentences[new_section]:
                        available_sentences[new_section].remove(i)
            else:
                available_sentences[new_section].remove(new_sentence_index)

        return {'context': new_context, 'section': new_section}


def extract_sections(row):
    title = row['title']
    text = row['text']
    section = row['section'].split('<sep>')[0]
    depth = row['depth']
    page = {'title': title, 'section': section,
            'sentences': [], 'depth': depth}
    if section in ['Notes', 'References', 'Sources', 'External links', 'Further reading', 'Other websites', 'Sources and references']:
        return page
    text = text.strip()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r' +', ' ', text)
    sentences = [elem.strip() + '\n' for elem in text.split('\n')
                 if elem.strip() != '']
    if sentences:
        sentences[-1] = sentences[-1][:-1]
    for sentence in sentences:
        split_sentences = sent_tokenize(sentence)
        clean_split_sentences = []
        i = 0
        while i < len(split_sentences):
            if len(split_sentences[i]) < 10:
                if i > 0 and i < len(split_sentences) - 1:
                    clean_split_sentences[-1] += ' ' + \
                        split_sentences[i] + ' ' + split_sentences[i+1]
                    i += 2
                elif i == 0 and i < len(split_sentences) - 1:
                    clean_split_sentences.append(
                        split_sentences[i] + ' ' + split_sentences[i+1])
                    i += 2
                elif i > 0 and i == len(split_sentences) - 1:
                    clean_split_sentences[-1] += ' ' + split_sentences[i]
                    i += 1
                else:
                    clean_split_sentences.append(split_sentences[i])
                    i += 1
            else:
                clean_split_sentences.append(split_sentences[i])
                i += 1
        for s in clean_split_sentences:
            s = s.strip()
            s = re.sub(' +', ' ', s)
            s = re.sub('\n+', '\n', s)
            s = re.sub('\n +', '\n', s)
            if s:
                page['sentences'].append(s)
    return page


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
    parser.add_argument('--join_samples', action='store_true',
                        help='Join positive and its negative samples into one row')

    parser.set_defaults(join_samples=False)
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
        if mention == '':
            continue
        if title in entity_map:
            entity_map[title].add(mention)
        else:
            entity_map[title] = set([mention])
    for row in mention_map_2:
        title = unencode_title(row['target_title'])
        mention = row['mention']
        if mention == '':
            continue
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
    pool = Pool(20)
    for output in tqdm(pool.imap_unordered(extract_sections, df_sections_train), total=len(df_sections_train)):
        title = output['title']
        section = output['section']
        sentences = output['sentences']
        depth = output['depth']
        if not sentences:
            continue
        if title not in page_sections_train:
            page_sections_train[title] = {}
        if section not in page_sections_train[title]:
            page_sections_train[title][section] = {
                'depth': depth, 'sentences': []}
        page_sections_train[title][section]['sentences'].extend(sentences)

    page_sections_val = {}
    for output in tqdm(pool.imap_unordered(extract_sections, df_sections_val), total=len(df_sections_val)):
        title = output['title']
        section = output['section']
        sentences = output['sentences']
        depth = output['depth']
        if title not in page_sections_val:
            page_sections_val[title] = {}
        if section not in page_sections_val[title]:
            page_sections_val[title][section] = {
                'depth': depth, 'sentences': []}
        page_sections_val[title][section]['sentences'].extend(sentences)
    pool.close()
    pool.join()

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
        'neg_type': 'none',
        'depth': row['link_section_depth'],
    } for row in tqdm(df_links_train)]
    random.shuffle(positive_samples_train)

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
                'noise_strategy': row['noise_strategy'],
                'depth': row['link_section_depth'],
            })
        except:
            print(
                f"Couldn't find {row['target_title']} from {row['source_title']}")
    random.shuffle(positive_samples_val)

    if not args.max_train_samples:
        args.max_train_samples = len(
            positive_samples_train) * (1 + args.neg_samples_train)

    if not args.max_val_samples and not args.max_test_samples:
        args.max_val_samples = len(
            positive_samples_val) * (1 + args.neg_samples_val) // 2
        args.max_test_samples = len(
            positive_samples_val) * (1 + args.neg_samples_val) // 2
    elif not args.max_test_samples:
        args.max_test_samples = len(
            positive_samples_val) * (1 + args.neg_samples_val) - args.max_val_samples
    elif not args.max_val_samples:
        args.max_val_samples = len(
            positive_samples_val) * (1 + args.neg_samples_val) - args.max_test_samples

    if len(positive_samples_train) * (1 + args.neg_samples_train) > args.max_train_samples:
        positive_samples_train = random.sample(
            positive_samples_train, math.ceil(
                args.max_train_samples / (1 + args.neg_samples_train))
        )
    if len(positive_samples_val) * (1 + args.neg_samples_val) > (args.max_val_samples + args.max_test_samples):
        positive_samples_val = random.sample(
            positive_samples_val, math.ceil(
                (args.max_val_samples + args.max_test_samples) / (1 + args.neg_samples_val))
        )

    print('Generating negative samples')
    negative_samples_train = construct_negative_samples(
        positive_samples_train, args.neg_samples_train, page_sections_train)
    negative_samples_val = construct_negative_samples(
        positive_samples_val, args.neg_samples_val, page_sections_val)

    print('Saving data')
    if not args.join_samples:
        df_train = pd.DataFrame(
            positive_samples_train + negative_samples_train)
        df_train = df_train.sample(
            n=min(args.max_train_samples, len(df_train))).reset_index(drop=True)

        df_val_full = pd.DataFrame(
            positive_samples_val + negative_samples_val).sample(frac=1).reset_index(drop=True)
    else:
        full_samples_train = positive_samples_train.copy()
        for i, sample in enumerate(negative_samples_train):
            true_index = i // args.neg_samples_train
            rel_index = i % args.neg_samples_train
            keys = ['link_context', 'label', 'neg_type',
                    'noise_strategy', 'source_section']
            for key in keys:
                if key in sample:
                    full_samples_train[true_index][f'{key}_neg_{rel_index}'] = sample[key]
                else:
                    full_samples_train[true_index][f'{key}_neg_{rel_index}'] = None
        df_train = pd.DataFrame(full_samples_train)

        full_samples_val = positive_samples_val.copy()
        for i, sample in enumerate(negative_samples_val):
            true_index = i // args.neg_samples_val
            rel_index = i % args.neg_samples_val
            keys = ['link_context', 'label', 'neg_type',
                    'noise_strategy', 'source_section']
            for key in keys:
                if key in sample:
                    full_samples_val[true_index][f'{key}_neg_{rel_index}'] = sample[key]
                else:
                    full_samples_val[true_index][f'{key}_neg_{rel_index}'] = None
        df_val_full = pd.DataFrame(full_samples_val)

    df_val = df_val_full.sample(
        n=args.max_val_samples // (1 + args.neg_samples_val))
    df_test = df_val_full.drop(df_val.index).sample(
        n=args.max_test_samples // (1 + args.neg_samples_val))

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
