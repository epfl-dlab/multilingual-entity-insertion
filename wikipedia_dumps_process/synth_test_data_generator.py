import argparse
import os
import pandas as pd
import os
from glob import glob
from tqdm import tqdm
import random
import urllib
import re
from ast import literal_eval
import json
import gc
from multiprocessing import Pool
import psutil
import sys

def update_targets(target_name, redirect_map):
    counter = 0
    while target_name in redirect_map:
        target_name = redirect_map[target_name]
        counter += 1
        if counter > 10:
            break
    return target_name

def simplify_html(html):
    if html is None:
        return None
    if html == '':
        return ''
    return 'a'
    
def compare_pages(input):
    output = {page['page_title']: {'links': [], 'found': 0, 'not_found': 0} for page in input}
    
    for page in input:
        old_page = page['old_page']
        new_page = page['new_page']
        old_version = old_page[list(old_page.keys())[0]][0]['source_version']
        if new_page[list(new_page.keys())[0]][0]['source_version'] == old_version:
            continue
        for target_page in new_page:
            if target_page not in old_page:
                for mod_link in new_page[target_page]:
                    output[page['page_title']]['links'].append(mod_link)
                    output[page['page_title']]['links'][-1]['old_version'] = old_version
            else:
                used = set([])
                for mod_link in new_page[target_page]:
                    found = False
                    for i, old_link in enumerate(old_page[target_page]):
                        if old_link['mention'] == mod_link['mention'] and old_link['source_section'] == mod_link['source_section'] and i not in used:
                            used.add(i)
                            found = True
                            output[page['page_title']]['found'] += 1
                            break
                    if not found:
                        output[page['page_title']]['not_found'] += 1
                        output[page['page_title']]['links'].append(mod_link)
                        output[page['page_title']]['links'][-1]['old_version'] = old_version
    return output
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--first_month_dir', '-i1', type=str,
                        required=True, help='Path to the first month directory')
    parser.add_argument('--second_month_dir', '-i2', type=str,
                        required=True, help='Path to the second month directory')
    parser.add_argument('--first_month', type=str, required=True,
                        help='First month in the format YYYYMMDD')
    parser.add_argument('--output_dir', '-o', type=str,
                        required=True, help='Path to the output directory')
    parser.add_argument('--no_mask_perc', type=float, default=0.4,
                        help='Percentage of links where no masking is applied')
    parser.add_argument('--mask_mention_perc', type=float, default=0.2,
                        help='Percentage of links where mention masking is applied')
    parser.add_argument('--mask_sentence_perc', type=float, default=0.3,
                        help='Percentage of links where sentence masking is applied')
    parser.add_argument('--mask_paragraph_perc', type=float, default=0.1,
                        help='Percentage of links where paragraph masking is applied')
    parser.add_argument('--max_pages', type=int, default=100_000, help='Maximum number pages to consider')

    args = parser.parse_args()

    # check if input directories exist
    if not os.path.exists(args.first_month_dir):
        raise Exception('First month directory does not exist')
    if not os.path.exists(args.second_month_dir):
        raise Exception('Second month directory does not exist')
    
    # check if percentage arguments are valid
    if abs(args.no_mask_perc + args.mask_mention_perc + args.mask_sentence_perc + args.mask_paragraph_perc - 1) > 1e-5:
        raise Exception('The sum of the masking percentages should be 1')
    if args.no_mask_perc < 0 or args.no_mask_perc > 1:
        raise Exception('The no mask percentage should be between 0 and 1')
    if args.mask_mention_perc < 0 or args.mask_mention_perc > 1:
        raise Exception('The mask mention percentage should be between 0 and 1')
    if args.mask_sentence_perc < 0 or args.mask_sentence_perc > 1:
        raise Exception('The mask sentence percentage should be between 0 and 1')
    if args.mask_paragraph_perc < 0 or args.mask_paragraph_perc > 1:
        raise Exception('The mask paragraph percentage should be between 0 and 1')

    # check if output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # get all the link files in the month directories
    first_month_link_files = glob(os.path.join(args.first_month_dir, "links", "links_*"))
    second_month_link_files = glob(
        os.path.join(args.second_month_dir, "links", "links_*"))
    random.shuffle(first_month_link_files)
    random.shuffle(second_month_link_files)

    # get the page files for the second month
    second_month_page_files = glob(
        os.path.join(args.second_month_dir, "good_pages", "good_pages_*"))
    
    redirect_map = pd.read_parquet(os.path.join(args.second_month_dir, 'redirect_map.parquet'))
    redirect_map = redirect_map.to_dict()['redirect']
    
    old_data = {}
    new_data = {}
    new_pages = 0
    new_page_liks = 0
    new_links = []
    no_id_found = 0
    no_id_not_found = 0
    for i in (pbar := tqdm(range(len(second_month_link_files)))):
        df = pd.read_parquet(second_month_link_files[i])
        if 'date_modified' in df.columns:
            df = df[pd.to_datetime(df['date_modified'], format='%Y-%m-%dT%H:%M:%SZ') >= pd.to_datetime(args.first_month, format='%Y%m%d')]
        df['target_title'] = df['target_title'].apply(lambda x: update_targets(x, redirect_map))
        for column in df.columns:
            if 'index' in column:
                df[column] = df[column].apply(lambda x: int(x) if x == x else x)
        df = df.to_dict(orient='records')
        for row in df:
            if row['source_title'] not in new_data:
                new_data[row['source_title']] = {}
            if row['target_title'] not in new_data[row['source_title']]:
                new_data[row['source_title']][row['target_title']] = []
            new_data[row['source_title']][row['target_title']].append(row)
        del df
        pbar.set_description(f"{len(new_data)} source pages currently saved in new_data")
        gc.collect()
        if len(new_data) > args.max_pages:
            break
    
    new_page_titles = set(new_data.keys())
    for i in (pbar := tqdm(range(len(first_month_link_files)))):
        df = pd.read_parquet(first_month_link_files[i])
        df = df[df['source_title'].isin(new_page_titles)]
        df['target_title'] = df['target_title'].apply(lambda x: update_targets(x, redirect_map))
        for column in df.columns:
            if 'index' in column:
                df[column] = df[column].apply(lambda x: int(x) if x == x else x)
        df = df.to_dict(orient='records')
        for row in df:
            if row['target_title'] not in new_data[row['source_title']]:
                continue
            if row['source_title'] not in old_data:
                old_data[row['source_title']] = {}
            if row['target_title'] not in old_data[row['source_title']]:
                old_data[row['source_title']][row['target_title']] = []
            old_data[row['source_title']][row['target_title']].append(row)
        del df
        pbar.set_description(f"{len(old_data)} source pages currently saved in old_data")
        gc.collect()
        
    pool_input = []
    pages = []
    new_pages_titles = list(new_data.keys())
    for page in tqdm(new_pages_titles):
        if page not in old_data:
            continue
        pages.append({'old_page': old_data[page], 'new_page': new_data[page], 'page_title': page})
        if len(pages) == 1000:
            pool_input.append(pages)
            pages = []
    if len(pages) > 0:
        pool_input.append(pages)
    del pages
    gc.collect()
    
    with Pool(5) as p:
        for output in tqdm(p.imap_unordered(compare_pages, pool_input), total=len(pool_input)):
            for page in output:
                new_links.extend(output[page]['links'])
                no_id_found += output[page]['found']
                no_id_not_found += output[page]['not_found']
                del new_data[page]
                del old_data[page]
            gc.collect()
    del pool_input
    del new_data
    del old_data
    gc.collect()

    # print(
    #     f"The new data has {new_pages} new pages and {new_page_links} links in these new pages")
    print(f"There are {len(new_links)} new links in the new data")
    print(f"From the links without ID, {no_id_found} ({no_id_found / (no_id_found + no_id_not_found) * 100:.2f}%) were matched to old links.")

    print('Loading pages')
    good_page = set([])
    for file in tqdm(second_month_page_files):
        temp = pd.read_parquet(file, columns=['title'])
        good_page = good_page.union(set(temp['title'].tolist()))

    print('Cleaning the links')
    clean_links = []
    fail_target_id = 0
    fail_source_qid = 0
    fail_target_qid = 0
    fail_target_title = 0
    fail_source_title = 0
    fail_context = 0
    fail_selflink = 0
    for link in tqdm(new_links):
        if link['target_ID'] is None:
            fail_target_id += 1
            continue
        if link['source_QID'] is None:
            fail_source_qid += 1
            continue
        if link['source_title'] not in good_page:
            fail_source_title += 1
            continue
        if link['target_QID'] is None:
            fail_target_qid += 1
            continue
        if link['target_title'] not in good_page:
            fail_target_title += 1
            continue
        if link['target_title'] == link['source_title']:
            fail_selflink += 1
            continue
        if link['context'] is None:
            fail_context += 1
            continue
        link['context'] = "\n".join(
            line for line in link['context'].split("\n") if line.strip() != '')
        for key in link:
            if 'index' in key:
                link[key] = int(link[key])
        clean_links.append(link)

    print(
        f"Out of the {len(new_links)} new links, {len(clean_links)} ({len(clean_links) / len(new_links) * 100:.2f}%) are valid")
    print(f"\t- Fail target ID: {fail_target_id} ({fail_target_id / len(new_links) * 100:.2f}%)")
    print(f"\t- Fail source QID: {fail_source_qid} ({fail_source_qid / len(new_links) * 100:.2f}%)")
    print(f"\t- Fail target QID: {fail_target_qid} ({fail_target_qid / len(new_links) * 100:.2f}%)")
    print(f"\t- Fail bad target page: {fail_target_title} ({fail_target_title / len(new_links) * 100:.2f}%)")
    print(f"\t- Fail bad source page: {fail_source_title} ({fail_source_title / len(new_links) * 100:.2f}%)")
    print(f"\t- Fail context: {fail_context} ({fail_context / len(new_links) * 100:.2f}%)")
    print(f"\t- Fail selflink: {fail_selflink} ({fail_selflink / len(new_links) * 100:.2f}%)")

    del new_links
    gc.collect()

    print('Triaging links to know which corruptions can be applied')
    mask_paragraph_links = []
    mask_mention_links = []
    mask_sentence_links = []
    no_mask_links = []
    for link in clean_links:
        # mask span
        if (link['context'][:link['context_span_start_index']] + link['context'][link['context_span_end_index']:]).strip() != '':
            if link['context_span_start_index'] <= link['context_sentence_start_index'] and link['context_span_end_index'] >= link['context_sentence_end_index']:
                mask_paragraph_links.append(link)
                continue

        # mask sentence
        if (link['context'][:link['context_sentence_start_index']] + link['context'][link['context_sentence_end_index']:]).strip() != '':
            if link['context_sentence_start_index'] <= link['context_mention_start_index'] and link['context_sentence_end_index'] > link['context_mention_end_index'] + 1:
                mask_sentence_links.append(link)
                continue

        # mask mention
        if (link['context'][:link['context_mention_start_index']] + link['context'][link['context_mention_end_index']:]).strip() != '':
            mask_mention_links.append(link)
            continue

        # no mask
        no_mask_links.append(link)

    print("Applying the corruptions")
    final_links = []
    random.shuffle(mask_paragraph_links)
    if len(mask_paragraph_links) < args.mask_paragraph_perc * len(clean_links):
        print(
            f'Not possible to satisfy the {args.mask_paragraph_perc * 100}% paragraph masking request. Using {len(mask_paragraph_links) / len(clean_links) * 100:.2f}% instead')
        print(
            f'Increasing the mask sentence percentage to {args.mask_sentence_perc * 100 + (args.mask_paragraph_perc - len(mask_paragraph_links) / len(clean_links)) * 100:.2f}%')
        args.mask_sentence_perc += args.mask_paragraph_perc - \
            len(mask_paragraph_links) / len(clean_links)
    for link in mask_paragraph_links[:int(len(clean_links) * args.mask_paragraph_perc)]:
        mod_link = link.copy()
        mod_link['original_context'] = mod_link['context']
        mod_link['context'] = mod_link['context'][:int(
            mod_link['context_span_start_index'])] + mod_link['context'][int(mod_link['context_span_end_index']):]
        mod_link['context'] = re.sub(' +', ' ', mod_link['context'])
        mod_link['context'] = re.sub('\n ', '\n', mod_link['context'])
        mod_link['context'] = re.sub('\n+', '\n', mod_link['context'])
        mod_link['context'] = mod_link['context'].strip()
        mod_link['noise_strategy'] = 'mask_span'
        # current_links = literal_eval(mod_link['current_links'])
        # clean_contexts_links = {}
        # for target in current_links:
        #     if current_links[target]['region'] in ['sentence', 'span']:
        #         continue
        #     clean_contexts_links[target] = current_links[target]
        # mod_link['current_links'] = str(clean_contexts_links)
        final_links.append(mod_link)

    mask_sentence_links.extend(mask_paragraph_links[int(
        len(clean_links) * args.mask_paragraph_perc):])
    random.shuffle(mask_sentence_links)
    if len(mask_sentence_links) < args.mask_sentence_perc * len(clean_links):
        print(
            f'Not possible to satisfy the {args.mask_sentence_perc * 100}% sentence masking request. Using {len(mask_sentence_links) / len(clean_links) * 100:.2f}% instead')
        print(
            f'Increasing the mask mention percentage to {args.mask_mention_perc * 100 + (args.mask_sentence_perc - len(mask_sentence_links) / len(clean_links)) * 100:.2f}%')
        args.mask_mention_perc += args.mask_sentence_perc - \
            len(mask_sentence_links) / len(clean_links)
    for link in mask_sentence_links[:int(len(clean_links) * args.mask_sentence_perc)]:
        mod_link = link.copy()
        mod_link['original_context'] = mod_link['context']
        mod_link['context'] = mod_link['context'][:int(
            mod_link['context_sentence_start_index'])] + mod_link['context'][int(mod_link['context_sentence_end_index']):]
        mod_link['context'] = re.sub(' +', ' ', mod_link['context'])
        mod_link['context'] = re.sub('\n ', '\n', mod_link['context'])
        mod_link['context'] = mod_link['context'].strip()
        mod_link['noise_strategy'] = 'mask_sentence'
        # current_links = literal_eval(mod_link['current_links'])
        # clean_contexts_links = {}
        # for target in current_links:
        #     if current_links[target]['region'] in ['sentence']:
        #         continue
        #     clean_contexts_links[target] = current_links[target]
        # mod_link['current_links'] = str(clean_contexts_links)
        final_links.append(mod_link)

    mask_mention_links.extend(mask_sentence_links[int(
        len(clean_links) * args.mask_sentence_perc):])
    random.shuffle(mask_mention_links)
    if len(mask_mention_links) < args.mask_mention_perc * len(clean_links):
        print(
            f'Not possible to satisfy the {args.mask_mention_perc * 100}% mention masking request. Using {len(mask_mention_links) / len(clean_links) * 100:.2f}% instead')
        print(
            f'Increasing the no mask percentage to {args.no_mask_perc * 100 + (args.mask_mention_perc - len(mask_mention_links) / len(clean_links)) * 100:.2f}%')
        args.no_mask_perc += args.mask_mention_perc - \
            len(mask_mention_links) / len(clean_links)
    for link in mask_mention_links[:int(len(clean_links) * args.mask_mention_perc)]:
        mod_link = link.copy()
        mod_link['original_context'] = mod_link['context']
        mod_link['context'] = mod_link['context'][:int(
            mod_link['context_mention_start_index'])] + mod_link['context'][int(mod_link['context_mention_end_index']):]
        mod_link['context'] = re.sub(' +', ' ', mod_link['context'])
        mod_link['context'] = re.sub('\n ', '\n', mod_link['context'])
        mod_link['context'] = mod_link['context'].strip()
        mod_link['noise_strategy'] = 'mask_mention'
        final_links.append(mod_link)

    no_mask_links.extend(mask_mention_links[int(
        len(clean_links) * args.mask_mention_perc):])
    random.shuffle(no_mask_links)
    for link in no_mask_links:
        mod_link = link.copy()
        mod_link['original_context'] = mod_link['context']
        mod_link['context'] = re.sub(' +', ' ', mod_link['context'])
        mod_link['context'] = re.sub('\n ', '\n', mod_link['context'])
        mod_link['context'] = mod_link['context'].strip()
        mod_link['noise_strategy'] = 'no_mask'
        final_links.append(mod_link)

    print('In the end, we have the following distribution:')
    print(f"\t- Mask span: {len([link for link in final_links if link['noise_strategy'] == 'mask_span'])} ({len([link for link in final_links if link['noise_strategy'] == 'mask_span']) / len(final_links) * 100:.2f}%)")
    print(f"\t- Mask sentence: {len([link for link in final_links if link['noise_strategy'] == 'mask_sentence'])} ({len([link for link in final_links if link['noise_strategy'] == 'mask_sentence']) / len(final_links) * 100:.2f}%)")
    print(f"\t- Mask mention: {len([link for link in final_links if link['noise_strategy'] == 'mask_mention'])} ({len([link for link in final_links if link['noise_strategy'] == 'mask_mention']) / len(final_links) * 100:.2f}%)")
    print(f"\t- No mask: {len([link for link in final_links if link['noise_strategy'] == 'no_mask'])} ({len([link for link in final_links if link['noise_strategy'] == 'no_mask']) / len(final_links) * 100:.2f}%)")

    print("Saving data")
    df = pd.DataFrame(final_links)
    df.to_parquet(os.path.join(args.output_dir, 'val_links.parquet'))
