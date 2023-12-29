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
from ast import literal_eval
tqdm.pandas()

def unencode_title(title):
    clean_title = urllib.parse.unquote(title).replace('_', ' ')
    return clean_title


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_month1_dir', type=str,
                        required=True, help='Path to input directory for month 1')
    parser.add_argument('--input_month2_dir', type=str,
                        required=True, help='Path to input directory for month 2')
    parser.add_argument('--links_file', type=str,
                        required=True, help='Path to file for added links')
    parser.add_argument('--output_dir', type=str,
                        required=True, help='Output directory')
    parser.add_argument('--neg_samples_train', type=int,
                        default=1, help='Number of negative samples per positive sample for training')
    parser.add_argument('--neg_samples_val', type=int,
                        default=1, help='Number of negative samples per positive sample for validation')
    parser.add_argument('--max_train_samples', type=int, default=None,
                        help='Maximum number of train samples to generate')
    parser.add_argument('--max_val_samples', type=int, default=None,
                        help='Maximum number of validation samples to generate.')

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not os.path.exists(args.input_month1_dir):
        raise Exception(
            f'Input directory {args.input_month1_dir} does not exist')
    if not os.path.exists(args.input_month2_dir):
        raise Exception(
            f'Input directory {args.input_month2_dir} does not exist')
    if not os.path.exists(args.links_file):
        raise Exception(f'Links file {args.links_file} does not exist')

    print('Loading pages')
    page_files = glob(os.path.join(args.input_month1_dir, 'good_pages')) + \
        glob(os.path.join(args.input_month2_dir, 'good_pages'))
    page_files.sort()
    page_leads = {}
    for file in tqdm(page_files):
        df = pd.read_parquet(file, columns=['title', 'lead_paragraph'])
        df['title'] = df['title'].apply(unencode_title)
        df = df.to_dict(orient='records')
        for row in df:
            page_leads[row['title']] = row['lead_paragraph']

    print('Loading links')
    df_links = pd.read_parquet(args.links_file)
    df_links['source_title'] = df_links['source_title'].progress_apply(unencode_title)
    df_links['target_title'] = df_links['target_title'].progress_apply(unencode_title)

    print('Cleaning links')
    print(f'We started with {len(df_links)} links')
    print(df_links.columns)
    no_context = df_links['context'] == ''
    print(f'There are {no_context.sum()} links with no context')
    no_neg_contexts = df_links['negative_contexts'] == '[]'
    print(f'There are {no_neg_contexts.sum()} links with no negative contexts')
    missing_page = ~df_links['target_title'].isin(page_leads)
    print(f'There are {missing_page.sum()} links with missing pages')
    missing_section = df_links['missing_category'] == 'missing_section'
    print(f'There are {missing_section.sum()} links with missing sections')
    df_links = df_links[~no_context & ~no_neg_contexts &
                        ~missing_page & ~missing_section]
    print(f"After cleaning, there are {len(df_links)} links")
    df_links['target_lead'] = df_links['target_title'].apply(
        lambda x: page_leads[x])

    df_links = df_links.reset_index(drop=True)
    df_links = df_links.to_dict(orient='records')

    print('Loading mention map')
    mention_map = pd.read_parquet(os.path.join(
        args.input_month1_dir, 'mention_map.parquet'))
    mention_map_dict = mention_map.to_dict(orient='records')
    entity_map = {}
    for row in mention_map_dict:
        title = unencode_title(row['target_title'])
        mention = row['mention']
        if mention == '':
            continue
        if title in entity_map:
            entity_map[title].add(mention)
        else:
            entity_map[title] = set([mention])

    if not args.max_train_samples and not args.max_val_samples:
        args.max_train_samples = len(
            df_links) * 0.8 * (1 + args.neg_samples_train)
        args.max_val_samples = len(df_links) * 0.2 * (1 + args.neg_samples_val)
        print(
            f'Setting max_train_samples to {args.max_train_samples} and max_val_samples to {args.max_val_samples}')
    elif not args.max_train_samples:
        args.max_train_samples = (len(
            df_links) - args.max_val_samples // (1 + args.neg_samples_val)) * (1 + args.neg_samples_train)
        if args.max_train_samples < 0:
            args.max_train_samples = len(
                df_links) * 0.8 * (1 + args.neg_samples_train)
            args.max_val_samples = len(df_links) * \
                0.2 * (1 + args.neg_samples_val)
            print(
                f'Warning: max_val_samples is too large. Setting max_train_samples to {args.max_train_samples} and max_val_samples to {args.max_val_samples}')
    elif not args.max_val_samples:
        args.max_val_samples = (len(
            df_links) - args.max_train_samples // (1 + args.neg_samples_train)) * (1 + args.neg_samples_val)
        if args.max_val_samples < 0:
            args.max_train_samples = len(
                df_links) * 0.8 * (1 + args.neg_samples_train)
            args.max_val_samples = len(df_links) * \
                0.2 * (1 + args.neg_samples_val)
            print(
                f'Warning: max_train_samples is too large. Setting max_train_samples to {args.max_train_samples} and max_val_samples to {args.max_val_samples}')
    else:
        if args.max_train_samples // (1 + args.neg_samples_train) + args.max_val_samples // (1 + args.neg_samples_val) > len(df_links):
            # reduce their size while keeping the ratio
            ratio = args.max_train_samples // (1 + args.neg_samples_train) / \
                (args.max_train_samples // (1 + args.neg_samples_train) +
                 args.max_val_samples // (1 + args.neg_samples_val))
            args.max_train_samples = len(
                df_links) * ratio * (1 + args.neg_samples_train)
            args.max_val_samples = len(df_links) * \
                (1 - ratio) * (1 + args.neg_samples_val)
            print(
                f'Warning: max_train_samples and max_val_samples are too large. Setting max_train_samples to {args.max_train_samples} and max_val_samples to {args.max_val_samples}')

    print('Processing all contexts')
    all_contexts = []
    all_sections = []
    # all_current_links = []
    for i, link in tqdm(enumerate(df_links), total=len(df_links)):
        all_contexts.append(link['context'])
        all_sections.append(link['section'])
        # all_current_links.append(literal_eval(link['current_links']))
        df_links[i]['negative_contexts'] = literal_eval(link['negative_contexts'])
        for negative_context in df_links[i]['negative_contexts']:
            all_contexts.append(negative_context['context'])
            all_sections.append(negative_context['section'])
            # all_current_links.append(negative_context['current_links'])

    print('Generating positive and negative samples')
    random.shuffle(df_links)
    train_links = []
    val_links = []
    while len(train_links) < args.max_train_samples / (1 + args.neg_samples_train) and len(df_links) != 0:
        if len(train_links) % 1000 == 0:
            print(len(train_links), args.max_train_samples / (1 + args.neg_samples_train))
        link = df_links.pop()
        if link['target_title'] not in page_leads:
            continue
        # current_links = literal_eval(link['current_links'])
        # processed_current_links = {}
        # for current_link in current_links:
        #     title = unencode_title(current_link)
        #     if title not in page_leads:
        #         continue
        #     processed_current_links[title] = {'target_title': title,
        #                                       'target_lead': page_leads[title]}
        train_links.append({'source_title': link['source_title'],
                            'target_title': link['target_title'],
                            'source_lead': link['source_lead'],
                            'target_lead': page_leads[link['target_title']],
                            'link_context': link['context'],
                            'source_section': link['section'],
                            'missing_category': link['missing_category'] if link['missing_category'] is not None else 'present',
                            # 'current_links': str(processed_current_links)
                            })
        negative_contexts = link['negative_contexts']
        if len(negative_contexts) > args.neg_samples_train:
            negative_contexts = random.sample(
                negative_contexts, args.neg_samples_train)
            for i, negative_context in enumerate(negative_contexts):
                train_links[-1][f'source_section_neg_{i}'] = negative_context['section']
                train_links[-1][f'link_context_neg_{i}'] = negative_context['context']
                # current_links = negative_context['current_links']
                # processed_current_links = {}
                # for current_link in current_links:
                #     title = unencode_title(current_link)
                #     if title not in page_leads:
                #         continue
                #     processed_current_links[title] = {'target_title': title,
                #                                       'target_lead': page_leads[title]}
                # train_links[-1][f'current_links_neg_{i}'] = str(processed_current_links)
        else:
            for i, negative_context in enumerate(negative_contexts):
                train_links[-1][f'source_section_neg_{i}'] = negative_context['section']
                train_links[-1][f'link_context_neg_{i}'] = negative_context['context']
                # current_links = negative_context['current_links']
                # processed_current_links = {}
                # for current_link in current_links:
                #     title = unencode_title(current_link)
                #     if title not in page_leads:
                #         continue
                #     processed_current_links[title] = {'target_title': title,
                #                                       'target_lead': page_leads[title]}
                # train_links[-1][f'current_links_neg_{i}'] = str(processed_current_links)
            counter = len(negative_contexts)
            while f'source_section_neg_{args.neg_samples_train - 1}' not in train_links[-1]:
                index = random.randint(0, len(all_contexts) - 1)
                neg_context = all_contexts[index]
                neg_section = all_sections[index]
                # neg_current_links = all_current_links[index]
                if link['target_title'] not in entity_map:
                    entity_map[link['target_title']] = set(
                        [link['target_title']])
                for mention in entity_map[link['target_title']]:
                    if mention in neg_context:
                        continue
                train_links[-1][f'source_section_neg_{counter}'] = neg_section
                train_links[-1][f'link_context_neg_{counter}'] = neg_context
                # processed_neg_current_links = {}
                # for neg_current_link in neg_current_links:
                #     title = unencode_title(neg_current_link)
                #     if title not in page_leads:
                #         continue
                #     processed_neg_current_links[title] = {'target_title': title,
                #                                           'target_lead': page_leads[title]}
                # train_links[-1][f'current_links_neg_{counter}'] = str(processed_neg_current_links)
                counter += 1
    while len(val_links) < args.max_val_samples / (1 + args.neg_samples_val) and len(df_links) != 0:
        if len(val_links) % 1000 == 0:
            print(len(val_links), args.max_val_samples / (1 + args.neg_samples_val))
        link = df_links.pop()
        if link['target_title'] not in page_leads:
            continue
        # current_links = literal_eval(link['current_links'])
        # processed_current_links = {}
        # for current_link in current_links:
        #     title = unencode_title(current_link)
        #     if title not in page_leads:
        #         continue
        #     processed_current_links[title] = {'target_title': title,
        #                                       'target_lead': page_leads[title]}
        val_links.append({'source_title': link['source_title'],
                          'target_title': link['target_title'],
                          'source_lead': link['source_lead'],
                          'target_lead': page_leads[link['target_title']],
                          'link_context': link['context'],
                          'source_section': link['section'],
                          'missing_category': link['missing_category'] if link['missing_category'] is not None else 'present',
                        #   'current_links': str(processed_current_links)
                          })
        negative_contexts = link['negative_contexts']
        if len(negative_contexts) > args.neg_samples_val:
            negative_contexts = random.sample(
                negative_contexts, args.neg_samples_val)
            for i, negative_context in enumerate(negative_contexts):
                val_links[-1][f'source_section_neg_{i}'] = negative_context['section']
                val_links[-1][f'link_context_neg_{i}'] = negative_context['context']
                # current_links = negative_context['current_links']
                # processed_current_links = {}
                # for current_link in current_links:
                #     title = unencode_title(current_link)
                #     if title not in page_leads:
                #         continue
                #     processed_current_links[title] = {'target_title': title,
                #                                       'target_lead': page_leads[title]}
                # val_links[-1][f'current_links_neg_{i}'] = str(processed_current_links)
        else:
            for i, negative_context in enumerate(negative_contexts):
                val_links[-1][f'source_section_neg_{i}'] = negative_context['section']
                val_links[-1][f'link_context_neg_{i}'] = negative_context['context']
                # current_links = negative_context['current_links']
                # processed_current_links = {}
                # for current_link in current_links:
                #     title = unencode_title(current_link)
                #     if title not in page_leads:
                #         continue
                #     processed_current_links[title] = {'target_title': title,
                #                                       'target_lead': page_leads[title]}
                # val_links[-1][f'current_links_neg_{i}'] = str(processed_current_links)
            counter = len(negative_contexts)
            while f'source_section_neg_{args.neg_samples_val - 1}' not in val_links[-1]:
                index = random.randint(0, len(all_contexts) - 1)
                neg_context = all_contexts[index]
                neg_section = all_sections[index]
                # neg_current_links = all_current_links[index]
                if link['target_title'] not in entity_map:
                    entity_map[link['target_title']] = set(
                        [link['target_title']])
                for mention in entity_map[link['target_title']]:
                    if mention in neg_context:
                        continue
                val_links[-1][f'source_section_neg_{counter}'] = neg_section
                val_links[-1][f'link_context_neg_{counter}'] = neg_context
                # processed_neg_current_links = {}
                # for neg_current_link in neg_current_links:
                #     title = unencode_title(neg_current_link)
                #     if title not in page_leads:
                #         continue
                #     processed_neg_current_links[title] = {'target_title': title,
                #                                           'target_lead': page_leads[title]}
                # val_links[-1][f'current_links_neg_{counter}'] = str(processed_neg_current_links)
                counter += 1

    print('Saving data')
    df_train = pd.DataFrame(train_links)
    df_val = pd.DataFrame(val_links)

    # create directories if they don't exist
    if not os.path.exists(os.path.join(args.output_dir, 'train')):
        os.makedirs(os.path.join(args.output_dir, 'train'))
    if not os.path.exists(os.path.join(args.output_dir, 'val')):
        os.makedirs(os.path.join(args.output_dir, 'val'))
    df_train.to_parquet(os.path.join(
        args.output_dir, 'train', 'train.parquet'))
    df_val.to_parquet(os.path.join(args.output_dir, 'val', 'val.parquet'))

    # copy mention map to output directory
    mention_map.to_parquet(os.path.join(
        args.output_dir, 'mentions.parquet'))
